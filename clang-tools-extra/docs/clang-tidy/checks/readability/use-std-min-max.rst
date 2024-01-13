.. title:: clang-tidy - readability-use-std-min-max

readability-use-std-min-max
===========================

Replaces certain conditionals with ``std::min`` or ``std::max`` for readability,
promoting use of standard library functions. Note: This may impact
performance in critical code due to potential additional stores compared
to the original if statement.


Examples:

Before:

.. code-block:: c++

  void foo() {
    int a = 2, b = 3;
    if (a < b)
      a = b;
  }


After:

.. code-block:: c++

  void foo() {
    int a = 2, b = 3;
    a = std::max(a, b);
  }
