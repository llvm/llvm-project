.. title:: clang-tidy - misc-use-internal-linkage

misc-use-internal-linkage
=========================

Detects variables and functions that can be marked as static or moved into
an anonymous namespace to enforce internal linkage.

Static functions and variables are scoped to a single file. Marking functions
and variables as static helps to better remove dead code. In addition, it gives
the compiler more information and allows for more aggressive optimizations.

Example:

.. code-block:: c++

  int v1; // can be marked as static

  void fn1() {} // can be marked as static

  namespace {
    // already in anonymous namespace
    int v2;
    void fn2();
  }
  // already declared as extern
  extern int v2;

  void fn3(); // without function body in all declaration, maybe external linkage
  void fn3();

  // export declarations
  export void fn4() {}
  export namespace t { void fn5() {} }
  export int v2;

Options
-------

.. option:: FixMode

  Selects what kind of a fix the check should provide. The default is `UseStatic`.

  ``None``
    Don't fix automatically.

  ``UseStatic``
    Add ``static`` for internal linkage variable and function.
