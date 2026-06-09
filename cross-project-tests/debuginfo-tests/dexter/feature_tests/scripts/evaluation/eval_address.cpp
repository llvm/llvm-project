// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s | FileCheck %s

// Test evaluation of !address nodes in Dexter.

// CHECK:      Non-matching nodes:
// CHECK-SAME: Value(FalseStart)
// CHECK:      Non-matching nodes:
// CHECK-SAME: Value(EvenFalserStart)
// CHECK-NOT: Non-matching nodes

// CHECK: total_watched_steps: 12
// CHECK: correct_steps: 10
// CHECK: incorrect_steps: 2
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 2
// CHECK: seen_values: 11
// CHECK: missing_values: 2

struct SubRange {
  char *Begin;
  int Length;
};

int main() {
  char Data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  char *Start = Data;
  char *FalseStart = Data + 1;
  char *EvenFalserStart = Data + 2;
  char *Middle = Data + 5; // !dex_label begin
  char *NearEnd = Data + 8;
  char *Pos = Data + 4;
  for (int I = 0; I < 6; ++I) {
    Pos = Pos + 1; // !dex_label loop
  }
  SubRange Range = {Data + 2, 4};
  return 0; // !dex_label ret
}

/*
---
# `Start` will be correct and `FalseStart` will be incorrect, because `Start` is evaluated first.
!where {lines: !label begin}:
    !value Start: !address data
    !value FalseStart: !address data
# `EvenFalserStart` will also be incorrect, because it has been evaluated later.
!where {lines: !label begin + 1}:
    !value EvenFalserStart: !address data
!where {lines: !label loop}:
    !value Pos:
    - !address data + 4
    - !address data + 5
    - !address data + 6
    - !address data + 7
    - !address data + 8
    - !address data + 9
!where {lines: !label ret}:
    !value Middle: !address data + 5
    !value NearEnd: !address end - 2
    !value Range:
        Begin: !address data + 2
        Length: 4
...
*/
