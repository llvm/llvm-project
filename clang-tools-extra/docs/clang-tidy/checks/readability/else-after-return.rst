.. title:: clang-tidy - readability-else-after-return

readability-else-after-return
=============================

`LLVM Coding Standards <https://llvm.org/docs/CodingStandards.html>`_ advises to
reduce indentation where possible and where it makes understanding code easier.
Early exit is one of the suggested enforcements of that. Please do not use
``else`` or ``else if`` after something that interrupts control flow - like
``return``, ``break``, ``continue``, ``throw``.

The following piece of code illustrates how the check works. This piece of code:

.. code-block:: c++

    void foo(int Value) {
      int Local = 0;
      for (int i = 0; i < 42; i++) {
        if (Value == 1) {
          return;
        } else {
          Local++;
        }

        if (Value == 2)
          continue;
        else
          Local++;

        if (Value == 3) {
          throw 42;
        } else {
          Local++;
        }
      }
    }


Would be transformed into:

.. code-block:: c++

    void foo(int Value) {
      int Local = 0;
      for (int i = 0; i < 42; i++) {
        if (Value == 1) {
          return;
        }
        Local++;

        if (Value == 2)
          continue;
        Local++;

        if (Value == 3) {
          throw 42;
        }
        Local++;
      }
    }

Options
-------

.. option:: WarnOnUnfixable

   When `true`, emit a warning for cases where the check can't output a
   Fix-It. These can occur with declarations inside the ``else`` branch that
   would have an extended lifetime if the ``else`` branch was removed.
   Default value is `true`.

.. option:: WarnOnConditionVariables

   When `true`, the check will attempt to refactor a variable defined inside
   the condition of the ``if`` statement that is used in the ``else`` branch
   defining them just before the ``if`` statement. This can only be done if
   the ``if`` statement is the last statement in its parent's scope.
   Default value is `true`.


LLVM alias
----------

There is an alias of this check called llvm-else-after-return.
In that version the options :option:`WarnOnUnfixable` and
:option:`WarnOnConditionVariables` are both set to `false` by default.

This check helps to enforce this `LLVM Coding Standards recommendation
<https://llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return>`_.
