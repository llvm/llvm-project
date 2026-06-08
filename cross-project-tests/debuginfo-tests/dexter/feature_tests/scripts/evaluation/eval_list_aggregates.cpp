// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s | FileCheck %s

/// Check that the debugger is able to correctly evaluate a list of expected
/// aggregate values.

struct Point {
    int X;
    int Y;
};

int main() {
  Point P { 1, 2 };
  P.X = 3; // !dex_label start
  P.Y = 0;
  P.X = 1;
  P.Y = 2;
  P = {0, 0};
  return 0; // !dex_label end
}

// CHECK: total_watched_steps: 6
// CHECK: correct_steps: 6
// CHECK: incorrect_steps: 0
// CHECK: partial_step_correctness: 6.0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: correct_step_coverage: 100.0% (6/6)
// CHECK: seen_values: 12
// CHECK: missing_values: 0

/*
---
!where {lines: !range [!label start, !label end]}:
    !value P:
        - X: 1
          Y: 2
        - X: 3
          Y: 2
        - X: 3
          Y: 0
        - X: 1
          Y: 0
        - X: 1
          Y: 2
        - X: 0
          Y: 0
...
*/
