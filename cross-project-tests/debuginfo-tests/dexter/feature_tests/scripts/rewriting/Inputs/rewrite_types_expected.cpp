// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} %S/Inputs/rewrite_types_expected.cpp

/// Test that Dexter can rewrite types, for individual variables and for all
/// variables in a scope.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.

// CHECK: Rewrote script to add 6 expected values.

// CHECK: correct_step_coverage: 100.0%
// CHECK: seen_types: 12
// CHECK: missing_types: 0

using NormalInt = int;

template <typename T> struct GenericDouble {
  T First;
  T Second;
};

template <typename T, typename U> struct Twople {
  T First;
  U Second;
};

struct NestedStruct {
  GenericDouble<int> IntMembers;
  Twople<Twople<float, NormalInt>, bool> ManyMembers;
};

GenericDouble<bool> GlobalDouble = {false, true};

int main() {
  int a = 0;
  NormalInt b = 1;
  Twople<bool, bool> c = {true, false};
  NestedStruct d = {{2, 4}, {{10, 11}, false}};
  auto e = false;
  return 0; // !dex_label ret
}

/*
---
? !where {lines: !label 'ret'}
: !type 'a': int
  !type 'b': NormalInt
  !type 'c':
    First: bool
    Second: bool
  !type 'd':
    IntMembers:
      First: int
      Second: int
    ManyMembers:
      First:
        First: float
        Second: int
      Second: bool
  !type 'e': bool
  !type 'GlobalDouble':
    First: bool
    Second: bool
...
*/
