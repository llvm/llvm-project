.. title:: clang-tidy - fuchsia-default-arguments-declarations

fuchsia-default-arguments-declarations
======================================

Warns if a function or method is declared with default parameters.

For example, the declaration:

.. code-block:: c++

  int foo(int value = 5) { return value; }

will cause a warning.

See the features disallowed in Fuchsia at https://fuchsia.googlesource.com/zircon/+/master/docs/cxx.md
