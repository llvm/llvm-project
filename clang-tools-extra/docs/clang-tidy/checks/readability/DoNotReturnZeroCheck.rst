.. title:: clang-tidy - readability-DoNotReturnZeroCheck

readability-DoNotReturnZeroCheck
================================

This Check warns about redundant return statements returning zero.

Before:

.. code-block:: c++

  int main(){
    int a;
    return 0;
  }


After:

.. code-block:: c++

  int main(){
    int a;

  }
