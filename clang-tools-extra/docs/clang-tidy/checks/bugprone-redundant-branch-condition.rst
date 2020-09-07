.. title:: clang-tidy - bugprone-redundant-branch-condition

bugprone-redundant-branch-condition
===================================

Finds condition variables in nested ``if`` statements that were also checked in
the outer ``if`` statement and were not changed.

Simple example:

.. code-block:: c

  bool onFire = isBurning();
  if (onFire) {
    if (onFire)
      scream();
  }

Here `onFire` is checked both in the outer ``if`` and the inner ``if`` statement
without a possible change between the two checks. The check warns for this code
and suggests removal of the second checking of variable `onFire`.

The checker also detects redundant condition checks if the condition variable
is an operand of a logical "and" (``&&``) or a logical "or" (``||``) operator:

.. code-block:: c

  bool onFire = isBurning();
  if (onFire) {
    if (onFire && peopleInTheBuilding > 0)
      scream();
  }

.. code-block:: c

  bool onFire = isBurning();
  if (onFire) {
    if (onFire || isCollapsing())
      scream();
  }

In the first case (logical "and") the suggested fix is to remove the redundant
condition variable and keep the other side of the ``&&``. In the second case
(logical "or") the whole ``if`` is removed similarily to the simple case on the
top.

The condition of the outer ``if`` statement may also be a logical "and" (``&&``)
expression:

.. code-block:: c

  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      if (onFire)
        scream();
    }
  }

The error is also detected if both the outer statement is a logical "and"
(``&&``) and the inner statement is a logical "and" (``&&``) or "or" (``||``).
The inner ``if`` statement does not have to be a direct descendant of the outer
one.

No error is detected if the condition variable may have been changed between the
two checks:

.. code-block:: c

  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (onFire && peopleInTheBuilding > 0)
      scream();
  }

Every possible change is considered, thus if the condition variable is not
a local variable of the function, it is a volatile or it has an alias (pointer
or reference) then no warning is issued.

Known limitations
^^^^^^^^^^^^^^^^^

The ``else`` branch is not checked currently for negated condition variable:

.. code-block:: c

  bool onFire = isBurning();
  if (onFire) {
    scream();
  } else {
    if (!onFire) {
      continueWork();
    }
  }

The checker currently only detects redundant checking of single condition
variables. More complex expressions are not checked:

.. code-block:: c

  if (peopleInTheBuilding == 1) {
    if (peopleInTheBuilding == 1) {
      doSomething();
    }
  }
