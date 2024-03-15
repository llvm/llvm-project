.. title:: clang-tidy - readability-math-missing-parentheses

readability-math-missing-parentheses
====================================

Check for mising parantheses in mathematical expressions that involve operators
of different priorities.

Before:

.. code-block:: c++

  int x = 1 + 2 * 3 - 4 / 5;


After:

.. code-block:: c++

  int x = (1 + (2 * 3)) - (4 / 5);