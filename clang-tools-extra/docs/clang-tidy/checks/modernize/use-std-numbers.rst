.. title:: clang-tidy - modernize-use-std-numbers

modernize-use-std-numbers
=========================

Finds constants and function calls to math functions that can be replaced
with c++20's mathematical constants from the ``numbers`` header and offers fix-it hints.
Does not match the use of variables or macros with that value, and instead, offers a replacement
at the definition of said variables and macros.

.. code-block:: c++
    double sqrt(double);
    double log(double);

    #define MY_PI 3.1415926  // #define MY_PI std::numbers::pi

    void foo() {
        const double Pi = 3.141592653589;  // const double Pi = std::numbers::pi
        const auto Use = Pi / 2;           // no match for Pi
        static constexpr double Euler = 2.7182818; // static constexpr double Euler = std::numbers::e;

        log2(exp(1));     // std::numbers::log2e;
        log2(Euler);      // std::numbers::log2e;
        1 / sqrt(MY_PI);  // std::numbers::inv_sqrtpi;
    }
