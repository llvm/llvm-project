.. title:: clang-tidy - bugprone-multiple-statement-macro

bugprone-multiple-statement-macro
=================================

Detect multiple statement macros that are used in unbraced conditionals. Only
the first statement of the macro will be inside the conditional and the other
ones will be executed unconditionally.

Example:

.. code-block:: c++

  #define INCREMENT_TWO(x, y) (x)++; (y)++
  if (do_increment)
    INCREMENT_TWO(a, b);  // (b)++ will be executed unconditionally.
