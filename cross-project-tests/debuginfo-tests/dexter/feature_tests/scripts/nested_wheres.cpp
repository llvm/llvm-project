// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s \
// RUN:   | FileCheck %s

/// Test that we correctly interpret nested !where+!and nodes during debugging
/// and evaluation.

int fib(int n) {
  if (n <= 1) {
    return 1;
  }
  int first = fib(n - 2);
  int second = fib(n - 1);
  return first + second;
}

int main() { return fib(4); }

// CHECK: correct_steps: 9
// CHECK: missing_values: 0

/*
---
!where {function: main}:
    !where {function: fib}:
        !and {lines: 9}:
            !value n: 4
        !where {function: fib}:
            !and {lines: 9}:
                !value n: [2, 3]
            !where {function: fib}:
                !and {lines: 9}:
                    !value n: [0, 1, 1, 2]
                !where {function: fib}:
                    !and {lines: 9}:
                        !value n: [0, 1]
...
*/
