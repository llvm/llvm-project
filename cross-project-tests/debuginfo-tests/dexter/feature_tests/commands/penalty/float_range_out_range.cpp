// Purpose:
//      Check that a \DexExpectWatchValue float_range that is not large enough
//      detects unexpected watch values.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: float_range_out_range.cpp:

int main() {
  float a = 1.0f;
  a = a - 0.5f;
  return a;  //DexLabel('check')
}

// DexExpectWatchValue('a', '1.00000', from_line=ref('check1'), to_line=ref('check2'), float_range=0.4)
