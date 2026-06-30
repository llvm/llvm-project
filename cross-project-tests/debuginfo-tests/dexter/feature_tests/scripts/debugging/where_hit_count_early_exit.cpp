// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t \
// RUN:   --timeout-total 10 -- %s 2>&1 | FileCheck %s

/// Test that when all root !where nodes have expired, we exit without waiting
/// for the debuggee to finish.

// CHECK-NOT: timeout reached

// CHECK-LABEL: Step 0
// CHECK-COUNT-3: getRandomNumber
// CHECK-NOT: getRandomNumber

/// All on one line for simplicity so that we only get one step per call.
int getRandomNumber(int Max) { return 4 % Max; }

int main() {
  // Bogo search
  int List[] = {0, 0, 0, 0, 5, 0, 0, 0, 0, 0};
  int SearchTarget = 0;
  while (true) {
    int NextSearch = getRandomNumber(10);
    if (List[NextSearch] == SearchTarget)
      return NextSearch;
  }
}

/*
---
!where {function: getRandomNumber, for_hit_count: 3}:
    !value Max: 10
...
*/
