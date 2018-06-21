.. title:: clang-tidy - bugprone-terminating-continue

bugprone-terminating-continue
=============================

Detects `do while` loops with a condition always evaluating to false that
have a `continue` statement, as this `continue` terminates the loop
effectively.

.. code-block:: c++

  void f() {
  do {
  	// some code
    continue; // terminating continue
    // some other code
  } while(false);
