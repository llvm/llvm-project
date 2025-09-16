// Purpose:
//      Check that \DexExpectWatchValue float_range=0.0 matches exact values.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: float_range_zero_match.cpp:

int main() {
  float a = 1.0f;
  return a; //DexLabel('check')
}

// DexExpectWatchValue('a', '1.0000000', on_line=ref('check'), float_range=0.0)
