// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-FORWARD
// RUN: %dexter_regression_test_cxx_build %s -o %t -DREVERSE_TEST
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-REVERSE

/// Check that the debugger is able to correctly evaluate lists of values for
/// individual members of aggregates, and that doing so removes ordering
/// constraints. in the first run P goes through the values:
///   [{1, 2}, {3, 2}, {5, 2}, {5, 4}, {5, 6}, {7, 7}]
/// In the second run P goes through the values:
///   [{1, 2}, {1, 4}, {1, 6}, {3, 6}, {5, 6}, {7, 7}]
/// Despite each run containing combinations of values that are not seen in the
/// other, the test should pass, as we check the sequence of X and Y values
/// individually rather than as a pair.

struct Point {
  int X;
  int Y;
};

struct Rectangle {
  Point TopLeft;
  Point Size;
};

int main() {
  Rectangle R{Point{1, 2}, Point{4, 4}};
  // !dex_label start
#ifndef REVERSE_TEST
  R.TopLeft.X = 3;
  R.TopLeft.X = 5;
  R.Size.X += 2;
  R.TopLeft.Y = 4;
  R.TopLeft.Y = 6;
#else
  R.TopLeft.Y = 4;
  R.TopLeft.Y = 6;
  R.Size.X += 2;
  R.TopLeft.X = 3;
  R.TopLeft.X = 5;
#endif
  R.TopLeft = {7, 7};
  R = {Point{0, 0}, Point{0, 0}};
  return 0; // !dex_label end
}

// CHECK-FORWARD: Step 0:
// CHECK-FORWARD:     Matching nodes:     [Value(R)={
// CHECK-FORWARD-SAME:  "TopLeft": { "X": 1, "Y": 2 },
// CHECK-FORWARD-SAME:  "Size": { "X": 4, "Y": 4 } }]
// CHECK-FORWARD: Step 1:
// CHECK-FORWARD:     Matching nodes:     [Value(R)={
// CHECK-FORWARD-SAME:  "TopLeft": { "X": 3, "Y": 2 },
// CHECK-FORWARD-SAME:  "Size": { "X": 4, "Y": 4 } }]
// CHECK-FORWARD: Step 2:
// CHECK-FORWARD:     Matching nodes:     [Value(R)={
// CHECK-FORWARD-SAME:  "TopLeft": { "X": 5, "Y": 2 },
// CHECK-FORWARD-SAME:  "Size": { "X": 4, "Y": 4 } }]
// CHECK-FORWARD: Step 3:
// CHECK-FORWARD:     Matching nodes:     [Value(R)={
// CHECK-FORWARD-SAME:  "TopLeft": { "X": 5, "Y": 2 },
// CHECK-FORWARD-SAME:  "Size": { "X": 6, "Y": 4 } }]
// CHECK-FORWARD: Step 4:
// CHECK-FORWARD:     Matching nodes:     [Value(R)={
// CHECK-FORWARD-SAME:  "TopLeft": { "X": 5, "Y": 4 },
// CHECK-FORWARD-SAME:  "Size": { "X": 6, "Y": 4 } }]
// CHECK-FORWARD: Step 5:
// CHECK-FORWARD:     Matching nodes:     [Value(R)={
// CHECK-FORWARD-SAME:  "TopLeft": { "X": 5, "Y": 6 },
// CHECK-FORWARD-SAME:  "Size": { "X": 6, "Y": 4 } }]
// CHECK-FORWARD: Step 6:
// CHECK-FORWARD:     Matching nodes:     [Value(R)={
// CHECK-FORWARD-SAME:  "TopLeft": { "X": 7, "Y": 7 },
// CHECK-FORWARD-SAME:  "Size": { "X": 6, "Y": 4 } }]

// CHECK-REVERSE: Step 0:
// CHECK-REVERSE:     Matching nodes:     [Value(R)={
// CHECK-REVERSE-SAME:  "TopLeft": { "X": 1, "Y": 2 },
// CHECK-REVERSE-SAME:  "Size": { "X": 4, "Y": 4 } }]
// CHECK-REVERSE: Step 1:
// CHECK-REVERSE:     Matching nodes:     [Value(R)={
// CHECK-REVERSE-SAME:  "TopLeft": { "X": 1, "Y": 4 },
// CHECK-REVERSE-SAME:  "Size": { "X": 4, "Y": 4 } }]
// CHECK-REVERSE: Step 2:
// CHECK-REVERSE:     Matching nodes:     [Value(R)={
// CHECK-REVERSE-SAME:  "TopLeft": { "X": 1, "Y": 6 },
// CHECK-REVERSE-SAME:  "Size": { "X": 4, "Y": 4 } }]
// CHECK-REVERSE: Step 3:
// CHECK-REVERSE:     Matching nodes:     [Value(R)={
// CHECK-REVERSE-SAME:  "TopLeft": { "X": 1, "Y": 6 },
// CHECK-REVERSE-SAME:  "Size": { "X": 6, "Y": 4 } }]
// CHECK-REVERSE: Step 4:
// CHECK-REVERSE:     Matching nodes:     [Value(R)={
// CHECK-REVERSE-SAME:  "TopLeft": { "X": 3, "Y": 6 },
// CHECK-REVERSE-SAME:  "Size": { "X": 6, "Y": 4 } }]
// CHECK-REVERSE: Step 5:
// CHECK-REVERSE:     Matching nodes:     [Value(R)={
// CHECK-REVERSE-SAME:  "TopLeft": { "X": 5, "Y": 6 },
// CHECK-REVERSE-SAME:  "Size": { "X": 6, "Y": 4 } }]
// CHECK-REVERSE: Step 6:
// CHECK-REVERSE:     Matching nodes:     [Value(R)={
// CHECK-REVERSE-SAME:  "TopLeft": { "X": 7, "Y": 7 },
// CHECK-REVERSE-SAME:  "Size": { "X": 6, "Y": 4 } }]

// CHECK: total_watched_steps: 8
// CHECK: correct_steps: 8
// CHECK: incorrect_steps: 0
// CHECK: partial_step_correctness: 8.0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: correct_step_coverage: 100.0% (8/8)
// CHECK: seen_values: 15
// CHECK: missing_values: 0

/*
---
!where {lines: !range [!label start, !label end]}:
  !value R:
    - TopLeft:
        - X: [1, 3, 5]
          Y: [2, 4, 6]
        - X: 7
          Y: 7
      Size:
        - X: [4, 6]
          Y: 4
    - TopLeft:
        X: 0
        Y: 0
      Size:
        X: 0
        Y: 0
...
*/
