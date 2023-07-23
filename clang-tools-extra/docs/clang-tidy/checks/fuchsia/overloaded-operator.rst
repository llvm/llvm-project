.. title:: clang-tidy - fuchsia-overloaded-operator

fuchsia-overloaded-operator
===========================

Warns if an operator is overloaded, except for the assignment (copy and move)
operators.

For example:

.. code-block:: c++

  int operator+(int);     // Warning

  B &operator=(const B &Other);  // No warning
  B &operator=(B &&Other) // No warning

See the features disallowed in Fuchsia at https://fuchsia.dev/fuchsia-src/development/languages/c-cpp/cxx?hl=en
