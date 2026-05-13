// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s | FileCheck %s

// Test evaluation of a Dexter test with no expects.

// CHECK: evaluate_nothing.cpp:
// CHECK: No expects found.

int main() {
    return 0;
}

/*
---
!where {lines: 10}: {} # No expects
...
*/
