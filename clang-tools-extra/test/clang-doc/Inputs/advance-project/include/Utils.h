#pragma once

/// Anonymous namespace for utility functions
namespace {
    /**
     * @brief Generate a "random" number
     *
     * @note This is not actually random in this implementation
     *
     * @return int A predetermined "random" number
     */
    int getRandomNumber();

    /**
     * @brief Helper function to convert int to string
     *
     * @param value The integer value to convert
     * @param buffer The char buffer to store the result
     * @param index Reference to the current index in the buffer
     */
    void intToString(int value, char* buffer, int& index);

    /**
     * @brief Helper function to convert double to string (simplified)
     *
     * @param value The double value to convert
     * @param buffer The char buffer to store the result
     * @param index Reference to the current index in the buffer
     */
    void doubleToString(double value, char* buffer, int& index);

    /**
     * @brief Helper function to print a string
     *
     * @param str The null-terminated string to print
     */
    void print(const char* str);
}