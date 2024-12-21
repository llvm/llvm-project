.. title:: clang-tidy - misc-redundant-expression

misc-redundant-expression
=========================

Detect redundant expressions which are typically errors due to copy-paste.

Depending on the operator expressions may be

- redundant,

- always ``true``,

- always ``false``,

- always a constant (zero or one).

Examples:

.. code-block:: c++

  ((x+1) | (x+1))             // (x+1) is redundant
  (p->x == p->x)              // always true
  (p->x < p->x)               // always false
  (speed - speed + 1 == 12)   // speed - speed is always zero
  int b = a | 4 | a           // identical expr on both sides
  ((x=1) | (x=1))             // expression is identical

Floats are handled except in the case that NaNs are checked like so:

.. code-block:: c++

  int TestFloat(float F) {
    if (F == F)               // Identical float values used
      return 1;
    return 0;
  }

  int TestFloat(float F) {
    // Testing NaN.
    if (F != F && F == F)     // does not warn
      return 1;
    return 0;
  }
