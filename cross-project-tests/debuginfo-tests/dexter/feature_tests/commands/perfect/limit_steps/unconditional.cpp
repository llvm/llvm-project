// Purpose:
//      Test that \DexLimitSteps can be used without a condition (i.e. the
//      breakpoint range is set any time from_line is stepped on).
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: unconditional.cpp

int glob;
int main() {
  int test = 0;
  for (test = 1; test < 4; test++) {
    glob += test; // DexLabel('from')
    glob += test; // DexLabel('to')
  }
  return test; // test = 4
}

// DexLimitSteps(from_line=ref('from'), to_line=ref('to'))
//// Unconditionally limit dexter's view of the program from line 'from' to
//// 'to'. Check for test=0, 1, 2 so that the test will fail if dexter sees
//// test=0 or test=4.
// DexExpectWatchValue('test', 1, 2, 3)

