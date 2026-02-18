.. title:: clang-tidy - modernize-use-constexpr

modernize-use-constexpr
=======================

Finds ``const`` local variables with literal types and compile-time
constant initializers that can be declared ``constexpr``.

Using ``constexpr`` makes the compile-time nature of the value
explicit and enables the compiler to enforce it.

For example:

.. code-block:: c++

  const int x = 42;            // -> constexpr int x = 42;
  const double d = 3.14;       // -> constexpr double d = 3.14;
  const bool b = true;         // -> constexpr bool b = true;
  const int s = sizeof(int);   // -> constexpr int s = sizeof(int);

The check only triggers when all of the following are true:

- The variable is a local variable (not global or static).
- The variable has ``const`` qualification (top-level).
- The type is a literal type.
- The initializer is a C++11 constant expression.
- The variable is not already ``constexpr``.
- The variable is not ``volatile``.
- The variable is not a reference or pointer type.
- The declaration is not in a macro.
