.. title:: clang-tidy - readability-math-missing-parentheses

readability-math-missing-parentheses
====================================

Check for missing parentheses in mathematical expressions that involve operators
of different priorities.

Parentheses in mathematical expressions clarify the order
of operations, especially with different-priority operators. Lengthy or multiline
expressions can obscure this order, leading to coding errors. IDEs can aid clarity
by highlighting parentheses. Explicitly using parentheses also clarifies what the 
developer had in mind when writing the expression. Ensuring their presence reduces
ambiguity and errors, promoting clearer and more maintainable code.

Before:

.. code-block:: c++

  int x = 1 + 2 * 3 - 4 / 5;


After:

.. code-block:: c++

  int x = 1 + (2 * 3) - (4 / 5);