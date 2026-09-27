.. title:: clang-tidy - readability-redundant-lambda-parameter-list

readability-redundant-lambda-parameter-list
===========================================

Finds lambda expressions with a redundant empty parameter list and removes it.

In C++11 and later, a lambda with no parameters does not require an explicit
``()`` unless it has a specifier such as ``mutable``, ``noexcept``, or a
trailing return type. In C++23 and later, ``()`` is redundant even when such
specifiers are present.

.. code-block:: c++

  // C++11 and later - the following lambdas will be rewritten:
  auto a = []() { return 42; };
  // becomes:
  auto a = [] { return 42; };

  auto b = [x = 1]() { return x; };
  // becomes:
  auto b = [x = 1] { return x; };

  // C++23 and later - the following lambdas will also be rewritten:
  auto c = []() mutable {};
  // becomes:
  auto c = [] mutable {};

  auto d = []() noexcept {};
  // becomes:
  auto d = [] noexcept {};

  auto e = []() -> int { return 0; };
  // becomes:
  auto e = [] -> int { return 0; };
