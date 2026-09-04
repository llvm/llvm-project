// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s \
// RUN:   | FileCheck %s

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

// CHECK: Step 0

/// While stopped at the first line of `factorial`, no !expect nodes should be
/// active.
// CHECK: simple_labels.cpp(8
// CHECK-NOT: Active !expect nodes
// CHECK: simple_labels.cpp(10

// CHECK: correct_step_coverage: 100.0%
// CHECK: seen_values: 9
// CHECK: missing_values: 0

/*
---
!where {lines: !label call}:
    !value a: 4
!where {function: factorial}:
    !and {lines: !range [!label factorial_start, !label factorial_end]}:
        !value result: [1, 2, 6, 24]
    !where {file: "Inputs/header.h", lines: !label mul}:
        !value result: [1, 2, 6, 24]
...
*/
