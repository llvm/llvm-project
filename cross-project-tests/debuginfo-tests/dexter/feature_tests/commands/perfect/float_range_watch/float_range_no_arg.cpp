// Purpose:
//      Check that omitted float_range from \DexExpectWatchValue turns off
//      the floating point range evalution and defaults back to
//      pre-float evalution.
//
// Since this test involves string comparison to the debugger output, it cannot
// work for both dbgeng and lldb, which output floats differently.
// UNSUPPORTED: system-darwin, system-windows
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: float_range_no_arg.cpp:

int main() {
  float a = 1.0f;
  return a;  //DexLabel('check')
}

// DexExpectWatchValue('a', '1', on_line=ref('check'))
