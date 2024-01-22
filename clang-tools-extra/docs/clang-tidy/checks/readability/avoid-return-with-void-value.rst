.. title:: clang-tidy - readability-avoid-return-with-void-value

readability-avoid-return-with-void-value
========================================

Finds return statements with ``void`` values used within functions with
``void`` result types.

A function with a ``void`` return type is intended to perform a task without
producing a return value. Return statements with expressions could lead
to confusion and may miscommunicate the function's intended behavior.

Example:

.. code-block::

   void g();
   void f() {
       // ...
       return g();
   }

In a long function body, the ``return`` statement suggests that the function
returns a value. However, ``return g();`` is a combination of two statements
that should be written as

.. code-block::

   g();
   return;

to make clear that ``g()`` is called and immediately afterwards the function 
returns (nothing).

In C, the same issue is detected by the compiler if the ``-Wpedantic`` mode
is enabled.

Options
-------

.. option::  IgnoreMacros

  The value `false` specifies that return statements expanded
  from macros are not checked. The default value is `true`.

.. option::  StrictMode

  The value `false` specifies that a direct return statement shall
  be excluded from the analysis if it is the only statement not 
  contained in a block like ``if (cond) return g();``. The default
  value is `true`.
