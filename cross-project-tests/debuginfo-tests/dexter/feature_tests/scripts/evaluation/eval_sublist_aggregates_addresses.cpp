// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s \
// RUN:   | FileCheck %s

/// Tests that expected value lists for aggregate members work as expected with
/// !address expected values.

/// Any of the expected values for First and Second could be selected for the
/// first step match, so the matching order for the first step should be:
/// 1. We attempt to match P.First->!address F, match succeeds and assigns
///    F = &Arr[2].
/// 2. We attempt to match P.Second->!address F, match fails as F == &Arr[2].
/// 3. We attempt to match P.Second->!address S, match succeeds and assigns
///    S = &Arr[4].
/// From there, the remaining values should resolve correctly.

struct PointerPair {
  char *First;
  char *Second;
};

void swapPtrs(char *&A, char *&B) {
  char *Tmp = A;
  A = B;
  B = Tmp;
}

int main() {
  char Arr[] = {0, 1, 2, 3, 4, 5, 6, 7};
  PointerPair P = {&Arr[2], &Arr[4]};
  // !dex_label start
  P.Second += 1;
  swapPtrs(P.First, P.Second);
  return 0; // !dex_label end
}

// CHECK: total_watched_steps: 3
// CHECK: correct_steps: 3
// CHECK: incorrect_steps: 0
// CHECK: partial_step_correctness: 3.0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: correct_step_coverage: 100.0% (3/3)
// CHECK: seen_values: 5
// CHECK: missing_values: 0

/*
---
!where {lines: !range [!label start, !label end]}:
    !value P:
        First: [!address F, !address S + 1]
        Second: [!address F, !address S, !address S + 1]
...
*/
