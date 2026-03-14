.. title:: clang-tidy - fuchsia-trailing-return

fuchsia-trailing-return
=======================

Functions that have trailing returns are disallowed, except for those using
``decltype`` specifiers and lambda with otherwise unutterable return types.

For example:

.. code-block:: c++

  // No warning
  int add_one(const int arg) { return arg; }

  // Warning
  auto get_add_one() -> int (*)(const int) {
    return add_one;
  }

Exceptions are made for lambdas and ``decltype`` specifiers:

.. code-block:: c++

  // No warning
  auto lambda = [](double x, double y) -> double {return x + y;};

  // No warning
  template <typename T1, typename T2>
  auto fn(const T1 &lhs, const T2 &rhs) -> decltype(lhs + rhs) {
    return lhs + rhs;
  }


See the features disallowed in Fuchsia at https://fuchsia.dev/fuchsia-src/development/languages/c-cpp/cxx?hl=en
