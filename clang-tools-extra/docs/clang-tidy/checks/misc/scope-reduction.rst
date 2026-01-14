.. title:: clang-tidy - misc-scope-reduction

misc-scope-reduction
====================

Detects local variables in functions whose scopes can be minimized.

Examples:

.. code-block:: cpp

    void test_deep_nesting() {
      int deep = 1; // 'deep' can be declared in a smaller scope
      if (true) {
        if (true) {
          if (true) {
            if (true) {
              int result = deep * 4;
            }
          }
        }
      }
    }

    void test_switch_case(int value) {
      int result = 0; // 'result' can be declared in a smaller scope
      switch (value) {
        case 1:
          result = 10;
          break;
        default:
          break;
      }
    }

    void test_for_loop_expressions() {
      int i; // 'i' can be declared in the for-loop initialization
      for (i = 0; i < 5; i++) {
        // loop body
      }
    }

Limitations
-----------

This checker cannot currently detect when a variable's previous value affects
subsequent iterations, resulting in false positives in some cases. This can
be addressed by implementing a pattern matcher that recognizes this
accumulator pattern across loop iterations or by using clang's builtin
Lifetime analysis.

.. code-block:: cpp

    void test_while_loop() {
      // falsely detects 'counter' can be moved to smaller scope
      int counter = 0;
      while (true) {
        counter++;
        if (counter > 10) break;
      }
    }

    void test_for_loop_reuse() {
      int temp = 0; // falsely detects 'temp' can be moved to smaller scope
      for (int i = 0; i<10; i++) {
        temp += i;
      }
    }

References
----------

This check corresponds to the CERT C Coding Standard rule.
`DCL19-C. Minimize the scope of variables and functions
<https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152335/DCL19-C.+Minimize+the+scope+of+variables+and+functions>`_.
