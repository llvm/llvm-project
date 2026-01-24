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

References
----------

This check corresponds to the CERT C Coding Standard rule.
`DCL19-C. Minimize the scope of variables and functions
<https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152335/DCL19-C.+Minimize+the+scope+of+variables+and+functions>`_.
