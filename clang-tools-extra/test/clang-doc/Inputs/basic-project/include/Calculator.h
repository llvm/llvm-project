#pragma once

/**
 * @brief A simple calculator class.
 *
 * Provides basic arithmetic operations.
 */
class Calculator {
public:
    /**
     * @brief Adds two integers.
     *
     * @param a First integer.
     * @param b Second integer.
     * @return int The sum of a and b.
     */
    int add(int a, int b);

    /**
     * @brief Subtracts the second integer from the first.
     *
     * @param a First integer.
     * @param b Second integer.
     * @return int The result of a - b.
     */
    int subtract(int a, int b);

    /**
     * @brief Multiplies two integers.
     *
     * @param a First integer.
     * @param b Second integer.
     * @return int The product of a and b.
     */
    int multiply(int a, int b);

    /**
     * @brief Divides the first integer by the second.
     *
     * @param a First integer.
     * @param b Second integer.
     * @return double The result of a / b.
     * @throw std::invalid_argument if b is zero.
     */
    double divide(int a, int b);

    /**
     * @brief Performs the mod operation on integers.
     *
     * @param a First integer.
     * @param b Second integer.
     * @return The result of a % b.
     */
    static int mod(int a, int b) {
      return a % b;
    }

    /**
     * @brief A static value.
     */
    static constexpr int static_val = 10;

    /**
     * @brief Holds a public value.
     */
    int public_val = -1;
};
