.. title:: clang-tidy - modernize-use-std-numbers

modernize-use-std-numbers
=========================

Finds constants and function calls to math functions that can be replaced
with C++20's mathematical constants from the ``numbers`` header and offers
fix-it hints.
Does not match the use of variables with that value, and instead,
offers a replacement for the definition of those variables.
Function calls that match the pattern of how the constant is calculated are
matched and replaced with the ``std::numbers`` constant.
The use of macros gets replaced with the corresponding ``std::numbers``
constant, instead of changing the macro definition.

The following list of constants from the ``numbers`` header are supported:

* ``e``
* ``log2e``
* ``log10e``
* ``pi``
* ``inv_pi``
* ``inv_sqrtpi``
* ``ln2``
* ``ln10``
* ``sqrt2``
* ``sqrt3``
* ``inv_sqrt3``
* ``egamma``
* ``phi``

The list currently includes all constants as of C++20.

The replacements use the type of the matched constant and can remove explicit
casts, i.e., switching between ``std::numbers::e``,
``std::numbers::e_v<float>`` and ``std::numbers::e_v<long double>`` where
appropriate.

.. code-block:: c++

    double sqrt(double);
    double log2(double);
    void sink(auto&&) {}
    void floatSink(float);

    #define MY_PI 3.1415926

    void foo() {
        const double Pi = 3.141592653589;           // const double Pi = std::numbers::pi
        const auto Use = Pi / 2;                    // no match for Pi
        static constexpr double Euler = 2.7182818;  // static constexpr double Euler = std::numbers::e;

        log2(exp(1));                               // std::numbers::log2e;
        log2(Euler);                                // std::numbers::log2e;
        1 / sqrt(MY_PI);                            // std::numbers::inv_sqrtpi;
        sink(MY_PI);                                // sink(std::numbers::pi);
        floatSink(MY_PI);                           // floatSink(std::numbers::pi);
        floatSink(static_cast<float>(MY_PI));       // floatSink(std::numbers::pi_v<float>);
    }

Options
-------

.. option:: DiffThreshold

    A floating point value that sets the detection threshold for when literals
    match a constant. A literal matches a constant if
    ``abs(literal - constant) < DiffThreshold`` evaluates to ``true``. Default
    is `0.001`.

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
