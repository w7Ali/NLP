import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import re
import nltk
from nltk.corpus import stopwords

# Check if stopwords are available, if not, download them
if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt')
if not nltk.data.find('corpora/stopwords'):
    nltk.download('stopwords')


# Initialize NLTK resources (you might not need this if you've already downloaded)
#nltk.download('punkt')
#nltk.download('stopwords')

st.title('Spam Detection App')

user_input = st.text_area('Enter a message:', '')

# Load the TF-IDF vectorizer and the model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('spam_model.pkl', 'rb') as f:
    spam_model = pickle.load(f)

if user_input:
    # Preprocess the user input
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', user_input)
    review = review.lower()
    review = [ps.stem(word) for word in review.split() if word not in set(stopwords.words('english'))]
    review = ' '.join(review)

    # Transform the preprocessed user input using the TF-IDF vectorizer
    user_input_tfidf = tfidf_vectorizer.transform([review])
    # Predict
    prediction = spam_model.predict(user_input_tfidf)

    if prediction[0]:
        st.write('Prediction: Spam')
    else:
        st.write('Prediction: Not Spam')
