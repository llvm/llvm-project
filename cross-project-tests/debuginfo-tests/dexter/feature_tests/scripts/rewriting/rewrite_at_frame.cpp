// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} \
// RUN:   %S/Inputs/rewrite_at_frame_expected.cpp

/// Tests that we can rewrite variables and scopes at frames above the current
/// frame.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.

int ChangeCount = 0;

void setVariable(int &Var, int NewValue) {
  Var = NewValue;
  ChangeCount += 1;
  return;
}

int main() {
  int X = 1;
  int Y = 2;
  int Z = 3;
  setVariable(X, 9);
  setVariable(Y, 8);
  setVariable(Z, 7);
  return 0;
}

// CHECK: total_watched_steps: 36
// CHECK: correct_steps: 36
// CHECK: incorrect_steps: 0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: seen_values: 10
// CHECK: missing_values: 0

/*
---
!where {function: setVariable}:
  !and {at_frame_idx: 1}:
    ? !value/all Locals
    ? !value ChangeCount
...
*/
