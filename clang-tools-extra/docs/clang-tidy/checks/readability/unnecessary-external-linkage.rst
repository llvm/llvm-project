.. title:: clang-tidy - readability-unnecessary-external-linkage

readability-unnecessary-external-linkage
========================================

Detects variable and function can be marked as static.

Static functions and variables are scoped to a single file. Marking functions
and variables as static helps to better remove dead code. In addition, it gives
the compiler more information and can help compiler make more aggressive
optimizations.

Example:

.. code-block:: c++
  int v1; // can be marked as static

  void fn1(); // can be marked as static

  namespace {
    // already in anonymous namespace
    int v2;
    void fn2();
  }
  // already declared as extern
  extern int v2;
