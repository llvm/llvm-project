.. title:: clang-tidy - bugprone-unhandled-code-paths

bugprone-unhandled-code-paths
=============================

This check discovers situations where code paths are not fully-covered.

``if-else if`` chains that miss a final ``else`` branch might lead to
unexpected program execution and be the result of a logical error.
If the missing ``else`` branch is intended you can leave it empty with
a clarifying comment.
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

Similar arguments hold for ``switch`` statements which do not cover all
possible code paths.

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

Options
-------

.. option:: WarnOnMissingElse

  Boolean flag that activates a warning for missing ``else`` branches.
  Default is `false`.
