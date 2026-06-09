// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --use-script --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} %S/Inputs/rewrite_aggregates_expected.cpp

/// Test that Dexter can write disaggregated expected values for aggregates,
/// including falling back to the parent value if sub_values contain errors,
/// e.g. for pointers that are not dereferencable.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.

// CHECK: Rewrote script to add 5 expected values.

// CHECK: total_watched_steps: 5
// CHECK: correct_steps: 5
// CHECK: incorrect_steps: 0
// CHECK: seen_values: 16
// CHECK: missing_values: 0

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
  Point P { 1, 2, 3 };
  int *I = &P.X;
  Rect R { { 1, 1, 1 }, { 2, 2, 2 } };
  int L[] = { 0, 1, 2, 3, 4 };
  int *InvalidPtr = nullptr;
  return 0; // !dex_label ret
}

/*
---
!where {lines: !label ret}:
    ? !value P
    ? !value I
    ? !value R
    ? !value L
    ? !value InvalidPtr
...
*/
