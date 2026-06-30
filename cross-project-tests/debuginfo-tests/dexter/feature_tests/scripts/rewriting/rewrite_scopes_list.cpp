// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --use-script --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} \
// RUN:   %S/Inputs/rewrite_scopes_list_expected.cpp

/// Tests that !value/all creates appropriate state nodes for the different
/// variables:
/// - TopFloor is live for the entire scope of the !value/all, so appears
///   directly under the root !where
/// - Elevator is live at two disjoint positions, so appears under two different
///   !and nodes.
/// - Ground and Button become live at the same time after the !value/all
///   becomes active and for the remainder of the program, so are both placed
///   under a single shared !and.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.

void ding() {}
void swapPassengers() {}

int main() {
  int TopFloor = 10;
  int Ground = 0, Button = 6; // !dex_label start
  for (int Elevator = Ground; Elevator < Button; ++Elevator) {
    ding();
  }
  swapPassengers();
  for (int Elevator = Button; Elevator < TopFloor; ++Elevator) {
    ding();
  }
  return 0; // !dex_label ret
}

// CHECK: Rewrote script to add 5 expected values.

// CHECK: total_watched_steps: 83
// CHECK: correct_steps: 83
// CHECK: incorrect_steps: 0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: seen_values: 13
// CHECK: missing_values: 0

/*
---
!where {lines: !range [!label start, !label ret]}:
    ? !value/all Locals
...
*/
