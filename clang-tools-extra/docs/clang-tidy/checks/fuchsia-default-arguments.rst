.. title:: clang-tidy - fuchsia-default-arguments

fuchsia-default-arguments
=========================

Warns if a function or method is declared or called with default arguments.

For example, the declaration:

.. code-block:: c++

  int foo(int value = 5) { return value; }

will cause a warning.

A function call expression that uses a default argument will be diagnosed.
Calling it without defaults will not cause a warning:

.. code-block:: c++

  foo();  // warning
  foo(0); // no warning

See the features disallowed in Fuchsia at https://fuchsia.googlesource.com/zircon/+/master/docs/cxx.md
