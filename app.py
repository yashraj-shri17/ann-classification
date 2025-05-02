import pandas as pd
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load model and preprocessing objects
model = tf.keras.models.load_model('model.keras')

with open('Label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('Onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit interface
st.title("Customer Churn Prediction")

# Input widgets
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
credit_score = st.number_input('Credit Score', 300, 850)
balance = st.number_input('Balance', 0.0)
estimated_salary = st.number_input('Estimated Salary', 0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create input DataFrame
input_data = pd.DataFrame({
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'CreditScore': [credit_score],
    'Balance': [balance],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# Process categorical features
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# One-hot encode geography using OneHotEncoder
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()

# Create a DataFrame for the one-hot encoded geography values
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=[f"Geography_{cat}" for cat in onehot_encoder_geo.categories_[0]]
)

# Combine features: Drop 'Geography' column and append encoded geography data
final_input = pd.concat([
    input_data.drop(['Geography'], axis=1),
    geo_encoded_df
], axis=1)

# Ensure correct column order based on the scaler's feature names
final_input = final_input[scaler.feature_names_in_]

# Scale and predict
input_scaled = scaler.transform(final_input)
prediction = model.predict(input_scaled)
probability = prediction[0][0]

# Display results
st.subheader("Prediction Result")
if probability > 0.5:
    st.error(f"High churn risk: {probability:.2%}")
else:
    st.success(f"Low churn risk: {probability:.2%}")
