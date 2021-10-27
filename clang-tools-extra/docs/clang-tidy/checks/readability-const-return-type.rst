.. title:: clang-tidy - readability-const-return-type

readability-const-return-type
=============================

Checks for functions with a ``const``-qualified return type and recommends
removal of the ``const`` keyword. Such use of `const` is usually superfluous,
and can prevent valuable compiler optimizations. Does not (yet) fix trailing
return types.

Examples:

.. code-block:: c++

  const int foo();
  const Clazz foo();
  Clazz *const foo();

Note that this applies strictly to top-level qualification, which excludes
pointers or references to const values. For example, these are fine:

.. code-block:: c++

   const int* foo();
   const int& foo();
   const Clazz* foo();
