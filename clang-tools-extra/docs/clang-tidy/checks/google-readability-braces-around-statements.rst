.. title:: clang-tidy - google-readability-braces-around-statements

google-readability-braces-around-statements
===========================================


Checks that bodies of ``if`` statements and loops (``for``, ``range-for``,
``do-while``, and ``while``) are inside braces

Before:

.. code:: c++

  if (condition)
    statement;

After:

.. code:: c++

  if (condition) {
    statement;
  }

Additionally, one can define an option ``ShortStatementLines`` defining the
minimal number of lines that the statement should have in order to trigger
this check.

The number of lines is counted from the end of condition or initial keyword
(``do``/``else``) until the last line of the inner statement.  Default value 0
means that braces will be added to all statements (not having them already).
