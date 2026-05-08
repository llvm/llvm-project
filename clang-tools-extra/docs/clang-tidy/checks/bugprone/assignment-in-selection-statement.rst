.. title:: clang-tidy - bugprone-assignment-in-selection-statement

bugprone-assignment-in-selection-statement
==========================================

Finds assignments within selection statements.
Such assignments may indicate programmer error because they may have been
intended as equality tests. The selection statements are conditions of ``if``
and loop (``for``, ``while``, ``do``) statements, condition of conditional
operator (``?:``) and any operand of a binary logical operator (``&&``,
``||``). The check finds assignments within these contexts if the single
expression is an assignment or the assignment is contained (recursively) in
last operand of a comma (``,``) operator or true and false expressions in a
conditional operator. The warning is suppressed if the assignment is placed in
extra parentheses, but only if the assignment is the single expression of a
condition (of ``if`` or a loop statement).

This check corresponds to the CERT rule
`EXP45-C. Do not perform assignments in selection statements
<https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152228/EXP45-C.+Do+not+perform+assignments+in+selection+statements>`_.

Examples
========

The check emits a warning in the following cases at the indicated locations:

.. code-block:: c++

  int x = 3;

  if (x = 4) // should it be `x == 4` instead of 'x = 4' ?
    x = x + 1;

  while ((x <= 11) || (x = 22)) // assignment appears as operand of a logical operator
    x += 2;

  do {
    x += 5;
  } while ((x > 10) ? (x = 11) : (x > 5)); // assignment in loop condition (from `x = 11`)

  for (int i = 0; i == 2, x = 5; ++i) // assignment in loop condition (from last operand of comma)
    foo1(i, x);

  for (int i = 0; i == 2, (x = 5); ++i) // assignment is not a single expression, parentheses do not prevent the warning
    foo1(i, x);

  int a = (x == 2) || (x = 3); // assignment appears in the operand a logical operator

The following cases do not produce a warning:

.. code-block:: c++

  if ((x = 1)) { // a single assignment between parentheses
    x += 10;

  if ((x = 1) != 0) { // assignment appears in a complex expression and without a logical operator
    ++x;

  if (foo(x = 9) && array[x = 8]) { // assignment appears in argument of function call or array index
    ++x;

  for (int i = 0; i = 2, x == 5; ++i) // assignment does not take part in the condition of the loop
    foo1(i, x);
