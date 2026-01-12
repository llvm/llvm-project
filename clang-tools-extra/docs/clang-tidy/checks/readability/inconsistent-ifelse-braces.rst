.. title:: clang-tidy - readability-inconsistent-ifelse-braces

readability-inconsistent-ifelse-braces
======================================

Detects ``if``/``else`` statements where one branch uses braces and the other
does not.

Before:

.. code-block:: c++

  if (condition) {
    statement;
  } else
    statement;

  if (condition)
    statement;

  if (condition)
    statement;
  else
    statement;

After:

.. code-block:: c++

  if (condition) {
    statement;
  } else {
    statement;
  }

  if (condition)
    statement;

  if (condition)
    statement;
  else
    statement;
