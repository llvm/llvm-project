.. title:: clang-tidy - readability-misplaced-array-index

readability-misplaced-array-index
=================================

This check warns for unusual array index syntax.

The following code has unusual array index syntax:

.. code-block:: c++

  void f(int *X, int Y) {
    Y[X] = 0;
  }

becomes

.. code-block:: c++

  void f(int *X, int Y) {
    X[Y] = 0;
  }

The check warns about such unusual syntax for readability reasons:
 * There are programmers that are not familiar with this unusual syntax.
 * It is possible that variables are mixed up.

