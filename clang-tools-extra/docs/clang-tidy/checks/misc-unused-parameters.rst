.. title:: clang-tidy - misc-unused-parameters

misc-unused-parameters
======================

Finds unused function parameters. Unused parameters may signify a bug in the
code (e.g. when a different parameter is used instead). The suggested fixes
either comment parameter name out or remove the parameter completely, if all
callers of the function are in the same translation unit and can be updated.

The check is similar to the `-Wunused-parameter` compiler diagnostic and can be
used to prepare a codebase to enabling of that diagnostic. By default the check
is more permissive (see :option:`StrictMode`).

.. code-block:: c++

  void a(int i) { /*some code that doesn't use `i`*/ }

  // becomes

  void a(int  /*i*/) { /*some code that doesn't use `i`*/ }

.. code-block:: c++

  static void staticFunctionA(int i);
  static void staticFunctionA(int i) { /*some code that doesn't use `i`*/ }

  // becomes

  static void staticFunctionA()
  static void staticFunctionA() { /*some code that doesn't use `i`*/ }

Options
-------

.. option:: StrictMode

   When zero (default value), the check will ignore trivially unused parameters,
   i.e. when the corresponding function has an empty body (and in case of
   constructors - no constructor initializers). When the function body is empty,
   an unused parameter is unlikely to be unnoticed by a human reader, and
   there's basically no place for a bug to hide.
