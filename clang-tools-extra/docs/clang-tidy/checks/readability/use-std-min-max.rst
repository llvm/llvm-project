.. title:: clang-tidy - readability-use-std-min-max

readability-use-std-min-max
===========================

Replaces certain conditional statements with equivalent ``std::min`` or ``std::max`` expressions, 
improving readability and promoting the use of standard library functions.
Note: While this transformation improves code clarity, it may not be
suitable for performance-critical code. Using ``std::min`` or ``std::max`` can
introduce additional stores, potentially impacting performance compared to
the original if statement that only assigns when the value needs to change.

Examples:

Before:

.. code-block:: c++

  void foo(){
    int a,b;
    if(a < b)
        a = b;
    }


After:

.. code-block:: c++

  void foo(){
    a = std::max(a, b);

  }
