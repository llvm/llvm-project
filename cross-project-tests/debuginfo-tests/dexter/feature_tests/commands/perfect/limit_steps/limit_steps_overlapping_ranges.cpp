// Purpose:
//      Ensure that multiple overlapping \DexLimitSteps ranges do not interfere.
//
// XFAIL: system-darwin
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: limit_steps_overlapping_ranges.cpp

int main() {
  int val1;
  int val2;
  int placeholder;
  for (int ix = 0; ix != 10; ++ix) {
    placeholder=val1+val2;   // DexLabel('from')
    if (ix == 0) {
      val1 = ix;
      val2 = ix;             // DexLabel('val1_check')
      placeholder=val1+val2; // DexLabel('val1_check_to')
    }
    else if (ix == 2) {
      val2 = ix;
      val1 = ix;             // DexLabel('val2_check')
      placeholder=val1+val2; // DexLabel('val2_check_to')
    }
    placeholder=val1+val2;   // DexLabel('to')
  }
  return val1 + val2;
}

// DexExpectWatchValue('ix', 0, 2, 5, from_line=ref('from'), to_line=ref('to'))
// DexExpectWatchValue('val1', 0, from_line=ref('val1_check'), to_line=ref('val1_check_to'))
// DexExpectWatchValue('val2', 2, from_line=ref('val2_check'), to_line=ref('val2_check_to'))

// DexLimitSteps('ix', 5, from_line=ref('from'), to_line=ref('to'))
// DexLimitSteps('val1', 0, from_line=ref('val1_check'), to_line=ref('val1_check_to'))
// DexLimitSteps('val2', 2, from_line=ref('val2_check'), to_line=ref('val2_check_to'))
