// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t \
// RUN:   -- %s | FileCheck %s --implicit-check-not="(int) -5"

// NB: Test CHECKs use line numbers, update them accordingly if adding/removing
//     lines in this test.

int flipIt(int Input) {
  int Result = -Input; // !dex_label flip_start
  return Result;       // !dex_label flip_ret
}

int main() {
  int Value = 5;
  Value = flipIt(Value);
  Value = flipIt(Value); // !dex_label second_call
  Value = flipIt(Value);
  return Value;
}

/// Test that when we can use !then under an at_frame_idx state node.
/// In the second call to flipIt, we trigger `!then step_out` before we reach
/// the line where we evaluate Input, and so we should only see Input=5.

// CHECK-LABEL: Step 0

// CHECK: flipIt(int)
// CHECK-NEXT: then_at_frame.cpp(9:
// CHECK-NOT: flipIt(int)

// CHECK: flipIt(int)
// CHECK-NEXT: then_at_frame.cpp(10:
// CHECK: Variables:
// CHECK: "Input": (int) 5
// CHECK-NOT: flipIt(int)

// CHECK: flipIt(int)
// CHECK-NEXT: then_at_frame.cpp(9:
// CHECK-NOT: flipIt(int)

// CHECK: flipIt(int)
// CHECK-NEXT: then_at_frame.cpp(9:
// CHECK-NOT: flipIt(int)

// CHECK: flipIt(int)
// CHECK-NEXT: then_at_frame.cpp(10:
// CHECK: Variables:
// CHECK: "Input": (int) 5
// CHECK-NOT: flipIt(int)

/*
---
!where {function: main}:
    !where {function: flipIt}:
        !and {lines: !label flip_ret}:
            !value Input: 5
        !and {lines: !label flip_start}:
            !and {at_frame_idx: 1, lines: !label second_call}: !then step_out
...
*/
