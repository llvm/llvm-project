.. title:: clang-tidy - readability-misleading-indentation

readability-misleading-indentation
==================================

Correct indentation helps to understand code. Mismatch of the syntactical
structure and the indentation of the code may hide serious problems.
Missing braces can also make it significantly harder to read the code,
therefore it is important to use braces.

The way to avoid dangling else is to always check that an ``else`` belongs
to the ``if`` that begins in the same column.

You can omit braces when your inner part of e.g. an ``if`` statement has only
one statement in it. Although in that case you should begin the next statement
in the same column with the ``if``.

Examples:

.. code-block:: c++

  // Dangling else:
  if (cond1)
    if (cond2)
      foo1();
  else
    foo2();  // Wrong indentation: else belongs to if(cond2) statement.

  // Missing braces:
  if (cond1)
    foo1();
    foo2();  // Not guarded by if(cond1).

Limitations
-----------

Note that this check only works as expected when the tabs or spaces are used
consistently and not mixed.
