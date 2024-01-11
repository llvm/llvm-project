.. title:: clang-tidy - readability-ConditionalToStdMinMax

readability-ConditionalToStdMinMax
==================================

Replaces certain conditional statements with equivalent std::min or std::max expressions, 
improving readability and promoting the use of standard library functions.

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