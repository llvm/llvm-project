// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s \
// RUN:   | FileCheck %s

/// Check that the debugger is able to evaluate the components of aggregate
/// values.

struct Point {
  int X;
  int Y;
  int Z;
};

struct Rect {
  Point TopLeft;
  Point BottomRight;
};

int main() {
  Point P{1, 2, 3};
  int *I = &P.X;
  Rect R{{1, 1, 1}, {2, 2, 2}};
  int L[] = {0, 1, 2, 3, 4};
  return 0; // !dex_label ret
}

// CHECK: total_watched_steps: 4
// CHECK: correct_steps: 3
// CHECK: incorrect_steps: 1
// CHECK: partial_step_correctness: 3.333
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 1
// CHECK: seen_values: 13
// CHECK: missing_values: 2

/*
---
!where {lines: !label ret}:
    !value P:
        X: 1 # Correct
        Y: 0 # Incorrect
        # Missing "Z"
        W: 8 # Not present
    !value I:
        "*I": 1
    !value R:
        TopLeft:
            X: 1
            Y: 1
            Z: 1
        BottomRight:
            X: 2
            Y: 2
            Z: 2
    !value L:
        "[0]": 0
        "[1]": 1
        "[2]": 2
        "[3]": 3
        "[4]": 4
...
*/
