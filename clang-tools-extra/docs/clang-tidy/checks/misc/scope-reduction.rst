.. title:: clang-tidy - misc-scope-reduction

misc-scope-reduction
===========================

Detects local variables in functions whose scopes can be minimized. This check
covers guidelines described by SEI DCL19-C, MISRA C++:2008 Rule 3-4-1, and MISRA
C:2012 Rule 8-9.

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

    void test_switch_multiple_cases(int value) {
      int accumulator = 0; // 'accumulator' can be declared in a smaller scope
      switch (value) {
        case 1:
          accumulator += 10;
          break;
        case 2:
          accumulator += 20;
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
This check corresponds to the CERT C Coding Standard rules
`DCL19-C. Minimize the scope of variables and functions
<https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152335/DCL19-C.+Minimize+the+scope+of+variables+and+functions>`_.
