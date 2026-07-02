// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s \
// RUN:   | FileCheck %s

// Test evaluation of !type nodes in Dexter.

// CHECK: correct_step_coverage: 100.0%
// CHECK: seen_types: 9
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
!where {lines: !label ret}:
  !type a: int
  !type b: NormalInt
  !type c: Twople<bool, bool>
  !type d:
    IntMembers:
      First: int
      Second: int
    ManyMembers:
      First:
        First: float
        Second: int
      Second: bool
  !type e: bool
...
*/
