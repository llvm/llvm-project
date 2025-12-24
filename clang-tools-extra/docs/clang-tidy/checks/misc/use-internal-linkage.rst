.. title:: clang-tidy - misc-use-internal-linkage

misc-use-internal-linkage
=========================

Detects variables, functions, and classes that can be marked as static or
moved into an anonymous namespace to enforce internal linkage.

Any entity that's only used within a single file should be given internal
linkage. Doing so gives the compiler more information, allowing it to better
remove dead code and perform more aggressive optimizations.

Example:

.. code-block:: c++

  int v1; // can be marked as static

  void fn1() {} // can be marked as static
  
  struct S1 {}; // can be moved into anonymous namespace

  namespace {
    // already in anonymous namespace
    int v2;
    void fn2();
    struct S2 {};
  }
  // already declared as extern
  extern int v2;

  void fn3(); // without function body in all declaration, maybe external linkage
  void fn3();

  // export declarations
  export void fn4() {}
  export namespace t { void fn5() {} }
  export int v2;
  export class C {};

Options
-------

.. option:: FixMode

  Selects what kind of a fix the check should provide. The default is `UseStatic`.

  - `None`
    Don't fix automatically.

  - `UseStatic`
    Add ``static`` for internal linkage variable and function.
