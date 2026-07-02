// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t -- %s | FileCheck %s --implicit-check-not="header.h(9"

/// Test that we can use file paths as part of a !where node, and that we can
/// use a trailing subset of the file path, including just the base filename, to
/// specify a file.

// CHECK-LABEL: Step 0
// CHECK: factorProduct
// CHECK: Inputs/header.h(4:13)
// CHECK:   "a": (int) 21
// CHECK-LABEL: Step 2
// CHECK: factorProduct
// CHECK: Inputs/header.h(11:10)
// CHECK:   "result": (int) 21
// CHECK-LABEL: Step 4
// CHECK: factorProduct
// CHECK: Inputs/header.h(4:13)
// CHECK:   "a": (int) 10
// CHECK-LABEL: Step 6
// CHECK: factorProduct
// CHECK: Inputs/header.h(11:10)
// CHECK:   "result": (int) 5

#include "Inputs/header.h"

int main() {
  int Result = factorProduct(21, 42);
  return Result + factorProduct(10, 35);
}

/*
---
!where {file: "header.h", lines: 4}:
    !value a: [21, 10]
!where {file: "Inputs/header.h", lines: 11}:
    !value result: [21, 5]
!where {file: "FakePath/header.h", lines: 9}:
    !value i: [3, 7, 5]
...
*/
