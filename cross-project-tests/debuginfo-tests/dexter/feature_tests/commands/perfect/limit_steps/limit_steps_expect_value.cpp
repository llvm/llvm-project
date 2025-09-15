// Purpose:
//      Ensure that limited stepping breaks for all expected values.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: limit_steps_expect_value.cpp

int main() {
  int i = 0;
  i = 1;    // DexLabel('from')
  i = 2;
  i = 3;
  return 0; // DexLabel('long_range')
}

// DexLimitSteps('i', '0', from_line=ref('from'), to_line=ref('long_range'))
// DexExpectWatchValue('i', 0, 1, 2, 3, from_line=ref('from'), to_line=ref('long_range'))
