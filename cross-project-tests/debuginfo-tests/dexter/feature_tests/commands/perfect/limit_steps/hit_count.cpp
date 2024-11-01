// Purpose:
//      Test that \DexLimitSteps keyword argument hit_count correctly limits
//      the number of times the command can trigger.
//
// XFAIL: system-darwin
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: hit_count.cpp

int a;
int main() {
  for (int i = 0; i < 4; i++) {
    a = i; // DexLabel('check')
  }
  return 0;
}

//// Unconditionally limit dexter's view of the program to 'on_line' and check
//// for i=0, i=1. The test will fail if dexter sees any other value for test.
// DexLimitSteps(hit_count=2, on_line=ref('check'))
// DexExpectWatchValue('i', '0', '1')
