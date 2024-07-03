#pragma once

/** @brief Maximum size for the IntArray typedef */
#define MAX_SIZE 100

/**
 * @brief Template class for a simple array
 *
 * @tparam T The type of elements in the array
 * @tparam Size The fixed size of the array
 */
template <typename T, int Size>
class Array {
public:
    /** @brief Default constructor */
    Array();

    /**
     * @brief Array access operator
     *
     * @param index The index of the element to access
     * @return T& Reference to the element at the given index
     */
    T& operator[](int index);

    /**
     * @brief Get the size of the array
     *
     * @return int The size of the array
     */
    int size() const;

private:
    T m_data[Size]; /**< The array data */
};