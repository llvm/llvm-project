.. title:: clang-tidy - bugprone-throw-keyword-missing

bugprone-throw-keyword-missing
==============================

Warns about a potentially missing ``throw`` keyword. If a temporary object is created, but the
object's type derives from (or is the same as) a class that has 'EXCEPTION', 'Exception' or
'exception' in its name, we can assume that the programmer's intention was to throw that object.

Example:

.. code-block:: c++

  void f(int i) {
    if (i < 0) {
      // Exception is created but is not thrown.
      std::runtime_error("Unexpected argument");
    }
  }


