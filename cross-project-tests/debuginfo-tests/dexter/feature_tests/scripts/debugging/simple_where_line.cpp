// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t -- %s | FileCheck %s

/// Test that we can perform a simple non-nested line-based !where.

// CHECK-LABEL: Step 0
// CHECK: multiply
// CHECK:   "a": (int) 7
// CHECK:   "b": (int) 6
// CHECK:   "result": (int) 42
// CHECK-LABEL: Step 2
// CHECK: main
// CHECK:   "a": (int) 6
// CHECK:   "b": (int) 7
// CHECK:   "c": (int) 42

int multiply(int b, int a) {
  int result = a * b;
  return result;
}

int main() {
  int a = 6;
  int b = 7;
  int c = multiply(a, b);
  return c;
}

/*
---
!where {lines: 19}:
    !value a: 7
    !value b: 6
    !value result: 42
!where {lines: 26}:
    !value a: 6
    !value b: 7
    !value c: 42
...
*/
