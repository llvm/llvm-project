// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --source-root-dir %S/Inputs \
// RUN:   --use-script --binary %t -- %s | FileCheck %s

// Check that when --source-root-dir is provided, labels will be checked
// relative to that directory.

#include "Inputs/header.h"

int factorial(int n) {
  int result = 1;
  // !dex_label factorial_start
  for (int i = 1; i <= n; ++i) {
    result = multiply(result, i);
  }
  return result; // !dex_label factorial_end
}

int main() {
  int a = 4;
  return factorial(a); // !dex_label call
}

// CHECK: total_watched_steps: 20
// CHECK: correct_steps: 20
// CHECK: incorrect_steps: 0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: seen_values: 9
// CHECK: missing_values: 0

/*
---
!where {lines: !label call}:
    !value a: 4
!where {function: factorial}:
    !and {lines: !range [!label factorial_start, !label factorial_end]}:
        !value result: [1, 2, 6, 24]
    !where {file: "header.h", lines: !label mul}:
        !value result: [1, 2, 6, 24]
...
*/
