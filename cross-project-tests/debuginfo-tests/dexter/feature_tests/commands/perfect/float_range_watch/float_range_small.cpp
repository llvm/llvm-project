// Purpose:
//      Check that \DexExpectWatchValue float_range=0.5 considers a range
//      difference of 0.49999 to be an expected watch value.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: float_range_small.cpp:

int main() {
  float a = 1.0f;
  a = a - 0.49999f;
  return a; //DexLabel('check')
}

// DexExpectWatchValue('a', '1.0', on_line=ref('check'), float_range=0.5)
