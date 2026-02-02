.. title:: clang-tidy - bugprone-assignment-in-selection-statement

bugprone-assignment-in-selection-statement
==========================================

Finds assignments within selection statements.
Such assignments may indicate programmer error because they may have been
intended as equality tests. The selection statements are conditions of ``if``
and loop (``for``, ``while``, ``do``) statements, condition of conditional
operator (``?:``) and any operand of a binary logical operator (``&&``, ``||``).
The check finds assignments within these contexts if the single expression is an
assignment or the assignment is contained (recursively) in last operand of a
comma (``,``) operator or true and false expressions in a conditional operator.
There is no warning if a single-standing assignment is enclosed in parentheses.

This check corresponds to the CERT rule
`EXP45-C. Do not perform assignments in selection statements
<https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152228/EXP45-C.+Do+not+perform+assignments+in+selection+statements>`_.

Examples
========

.. code-block:: c++

  int x = 3;

  if (x = 4) // warning: should it be `x == 4`?
    x = x + 1;

  if ((x = 1)) { // no warning: single assignment in parentheses
    x += 10;

  if ((x = 1) != 0) { // no warning: assignment appears in a complex expression and not with a logical operator
    ++x;

  if (foo(x = 9) && array[x = 8]) { // no warning: assignment appears in argument of function call or array index
    ++x;

  while ((x <= 11) || (x = 22)) // warning: the assignment is found as operand of a logical operator
    x += 2;

  do {
    x += 5;
  } while ((x > 10) ? (x = 11) : (x > 5)); // warning: assignment in loop condition (from `x = 11`)

  for (int i = 0; i == 2, x = 5; ++i) // warning: assignment in loop condition (from last operand of comma)
    foo1(i, x);

  for (int i = 0; i == 2, (x = 5); ++i) // warning: assignment is not a single expression, parentheses do not prevent the warning
    foo1(i, x);

  for (int i = 0; i = 2, x == 5; ++i) // no warning: assignment does not take part in the condition of the loop
    foo1(i, x);
  
  int a = (x == 2) || (x = 3); // warning: the assignment appears in the operand a logical operator
