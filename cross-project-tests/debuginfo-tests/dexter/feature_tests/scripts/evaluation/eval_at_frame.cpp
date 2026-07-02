// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s \
// RUN:   | FileCheck %s

// Test evaluation of !and{at_frame_idx} nodes in Dexter.

// CHECK: total_watched_steps: 18
// CHECK: correct_steps: 18
// CHECK: incorrect_steps: 0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: seen_values: 8
// CHECK: missing_values: 0

int Global = 4;

int bar(int Z) {
  Global *= 2;
  return Z / 2;
}

int foo(int Y) {
  int First = bar(Y);
  return First + bar(Y * 2) * 2; // !dex_label second_call
}

int main() {
  int X = 9;
  return foo(X + 1); // !dex_label root_call
}

/*
---
!where {function: main}:
  !where {function: foo}:
    !where {function: bar}:
      !value Z: [10, 20]
      !and {at_frame_idx: 1}:
        !value Y: 10
        !and {lines: !label second_call}:
            !value First: 5
      !and {at_frame_idx: 2, lines: !label root_call}:
        !value X: 9
        !value Global: [4, 8, 16]
...
*/
