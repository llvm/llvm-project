// Purpose:
//      Check that \DexExpectWatchValue float_range=0.0 matches only exact
//      values.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: float_range_zero_nonmatch.cpp:

int main() {
  float a = 1.0f;
  return a; //DexLabel('check')
}

// DexExpectWatchValue('a', '1.0000001', on_line=ref('check'), float_range=0.0)
