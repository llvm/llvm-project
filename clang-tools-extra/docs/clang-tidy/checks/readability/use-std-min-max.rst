.. title:: clang-tidy - readability-use-std-min-max

readability-use-std-min-max
===========================

Replaces certain conditional statements with equivalent ``std::min`` or ``std::max`` expressions, 
improving readability and promoting the use of standard library functions.
Note: this transformation may impact performance in performance-critical code due to potential 
additional stores compared to the original if statement.

Examples:

Before:

.. code-block:: c++

  void foo() {
    int a, b;
    if (a < b)
      a = b;
  }


After:

.. code-block:: c++

  void foo() {
    int a, b;
    a = std::max(a, b);
  }
