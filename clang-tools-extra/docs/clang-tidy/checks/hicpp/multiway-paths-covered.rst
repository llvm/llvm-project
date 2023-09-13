.. title:: clang-tidy - hicpp-multiway-paths-covered

hicpp-multiway-paths-covered
============================

This check discovers situations where code paths are not fully-covered.
It furthermore suggests using ``if`` instead of ``switch`` if the code will be more clear.
The `rule 6.1.2 <https://www.perforce.com/resources/qac/high-integrity-cpp-coding-standard/statements>`_
and `rule 6.1.4 <https://www.perforce.com/resources/qac/high-integrity-cpp-coding-standard/statements>`_
of the High Integrity C++ Coding Standard are enforced.

``if-else if`` chains that miss a final ``else`` branch might lead to unexpected
program execution and be the result of a logical error.
If the missing ``else`` branch is intended you can leave it empty with a clarifying
comment.
This warning can be noisy on some code bases, so it is disabled by default.

.. code-block:: c++

  void f1() {
    int i = determineTheNumber();

     if(i > 0) {
       // Some Calculation
     } else if (i < 0) {
       // Precondition violated or something else.
     }
     // ...
  }

Similar arguments hold for ``switch`` statements which do not cover all possible code paths.

.. code-block:: c++

  // The missing default branch might be a logical error. It can be kept empty
  // if there is nothing to do, making it explicit.
  void f2(int i) {
    switch (i) {
    case 0: // something
      break;
    case 1: // something else
      break;
    }
    // All other numbers?
  }

  // Violates this rule as well, but already emits a compiler warning (-Wswitch).
  enum Color { Red, Green, Blue, Yellow };
  void f3(enum Color c) {
    switch (c) {
    case Red: // We can't drive for now.
      break;
    case Green:  // We are allowed to drive.
      break;
    }
    // Other cases missing
  }


The `rule 6.1.4 <https://www.perforce.com/resources/qac/high-integrity-cpp-coding-standard/statements>`_
requires every ``switch`` statement to have at least two ``case`` labels other than a `default` label.
Otherwise, the ``switch`` could be better expressed with an ``if`` statement.
Degenerated ``switch`` statements without any labels are caught as well.

.. code-block:: c++

  // Degenerated switch that could be better written as `if`
  int i = 42;
  switch(i) {
    case 1: // do something here
    default: // do something else here
  }

  // Should rather be the following:
  if (i == 1) {
    // do something here
  }
  else {
    // do something here
  }


.. code-block:: c++

  // A completely degenerated switch will be diagnosed.
  int i = 42;
  switch(i) {}


Options
-------

.. option:: WarnOnMissingElse

  Boolean flag that activates a warning for missing ``else`` branches.
  Default is `false`.
