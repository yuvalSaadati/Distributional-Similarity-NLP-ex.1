import gensim.downloader as dl
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
# Word vectors
model = dl.load("word2vec-google-news-300")
vocab = model.index_to_key
#################################################################
#################################################################
print("Generating lists of the most 20 similar words according to word2vec:")
print(model.most_similar("foccacia_bread", topn=20))
print(model.most_similar("year", topn=20))
print(model.most_similar("citzen", topn=20))
print(model.most_similar("first", topn=20))
print(model.most_similar("in", topn=20))

#################################################################
#################################################################
print("./n 3 polysemous words such that the top-10 neighbors of each word reflect both word meanings:")
print("mouse: ", model.most_similar("mouse", topn=10))
print("head: ", model.most_similar("head", topn=10))
print("bat: ", model.most_similar("bat", topn=10))
print("./n 3 polysemous words such that the top-10 neighbors of each word reflect only a single meaning:")
print("bank: ", model.most_similar("bank", topn=10))
print("gallery: ", model.most_similar("gallery", topn=10))
print("park: ", model.most_similar("park", topn=10))

#################################################################
#################################################################
# # Specify the words for which you want to check similarity
word1 = "happy"
word2 = "joyful"
word3 = "sad"

# Check similarity between the two words
similarity_score1 = model.similarity(word1, word2)
similarity_score2 = model.similarity(word1, word3)

print("./n A triplet of words (word1, word2, word3) such that all of the following conditions hold:.\n ",
      "a) word1 and word2 are synonyms or almost synonyms ./n",
      "b) word1 and word3 are antonyms./n",
    "c) sim(word1,word2) < sim(word1, word3)")
print(similarity_score1)
print(similarity_score2)

#################################################################
#################################################################
print("./n The Effect of Different Corpora: ./n",
      "5 words whose top 10 neighbors based on the news corpus are very",
      "similar to their top 10 neighbors based on the twitter corpus:")
model_wiki = dl.load("glove-wiki-gigaword-200")
model_twitt = dl.load("glove-twitter-200")
print("weather: ", model_wiki.most_similar("weather",topn=10))
print("weather: ", model_twitt.most_similar("weather",topn=10))
print("apple: ", model_wiki.most_similar("apple",topn=10))
print("apple: ", model_twitt.most_similar("apple",topn=10))
print("nutrition: ", model_wiki.most_similar("nutrition",topn=10))
print("nutrition: ", model_twitt.most_similar("nutrition",topn=10))
print("football: ", model_wiki.most_similar("football",topn=10))
print("football: ", model_twitt.most_similar("football",topn=10))
print("car: ", model_wiki.most_similar("car",topn=10))
print("car: ", model_twitt.most_similar("car",topn=10))


print("./n 5 words who’s top 10 neighbors based on the",
      " news corpus which are substantially different",
      "from the top 10 neighbors based on the twitter corpus")
print("vector: ", model_wiki.most_similar("vector",topn=10))
print("vector: ", model_twitt.most_similar("vector",topn=10))
print("head: ", model_wiki.most_similar("head",topn=10))
print("head: ", model_twitt.most_similar("head",topn=10))
print("paper: ", model_wiki.most_similar("paper",topn=10))
print("paper: ", model_twitt.most_similar("paper",topn=10))
print("rose: ", model_wiki.most_similar("rose",topn=10))
print("rose: ", model_twitt.most_similar("rose",topn=10))
print("function: ", model_wiki.most_similar("function",topn=10))
print("function: ", model_twitt.most_similar("function",topn=10))

#################################################################
#################################################################
# Get the vocabulary
vocabulary = model.index_to_key[1:5000]
# for i, word in enumerate(vocabulary):
#     print(word)
vocabulary = model.index_to_key[1:5000]
# Filter words that end with "ed" or "ing"
ed_words = [word for word in vocabulary if word.endswith("ed")]
ing_words = [word for word in vocabulary if word.endswith("ing")]
# list of 708 words that end with "ed" or "ing"
filtered_words = ed_words + ing_words
# Create a matrix with 708 rows, each containing a 300-dim vector for one word
word_matrix_300D = np.zeros((len(filtered_words), 300))

for i, word in enumerate(filtered_words):
    try:
        word_matrix_300D[i, :] = model[word]
    except KeyError:
        # Handle the case where the word is not in the model's vocabulary
        print(f"Word '{word}' not found in the model's vocabulary.")
# reduce the dimensionality of the matrix to 2-d
pca = decomposition.PCA(n_components=2)
pca.fit(word_matrix_300D) 
word_matrix_2D = pca.transform(word_matrix_300D)

# Plot the vectors
# Color points that correspond to words that end with “ed” in blue, and points 
# that correspond to words that end with “ing” in green
plt.figure(figsize=(10, 6))
for i, word in enumerate(filtered_words):
    if word.endswith("ed"):
        plt.scatter(word_matrix_2D[i, 0], word_matrix_2D[i, 1], color='blue')
        # plt.text(word_matrix_2D[i, 0], word_matrix_2D[i, 1], word, fontsize=8, ha='right', va='bottom')
    else:
        plt.scatter(word_matrix_2D[i, 0], word_matrix_2D[i, 1], color='green')
        # plt.text(word_matrix_2D[i, 0], word_matrix_2D[i, 1], word, fontsize=8, ha='right', va='bottom')

# Set plot labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Visualization of Word Vectors')

# Show the plot
plt.show()

#################################################################
#################################################################
# Related words
print("./n Increase the number of neighbors from 20 to 100 in word2vec-google-news-300 model :")
foccacia_bread = model.most_similar("foccacia_bread", topn=100)
for word in foccacia_bread:
    print(word[0])
print(" ------------------- ")

first = model.most_similar("first", topn=100)
for word in first:
    print(word[0])
print(" ------------------- ")

