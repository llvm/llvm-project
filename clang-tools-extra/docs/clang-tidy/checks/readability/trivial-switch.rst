.. title:: clang-tidy - readability-trivial-switch

readability-trivial-switch
==========================

Finds trivial ``switch`` statements that can be written more clearly.

Every ``switch`` statement should have at least two ``case`` labels other than a ``default`` label.
Otherwise, the ``switch`` can be better expressed with an ``if`` statement.
``switch`` statements without any labels are diagnosed as well.

.. code-block:: c++

  int i = 42;

  switch (i) {
  case 1:
    doSomething();
    break;
  default:
    doSomethingElse();
    break;
  }

  // The switch can be written more clearly as:
  if (i == 1) {
    doSomething();
  } else {
    doSomethingElse();
  }

.. code-block:: c++

  // The switch without any labels will be diagnosed.
  int i = 42;
  switch (i) {}
