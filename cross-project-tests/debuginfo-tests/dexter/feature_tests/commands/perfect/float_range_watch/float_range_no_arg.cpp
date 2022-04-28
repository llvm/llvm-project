// Purpose:
//      Check that omitted float_range from \DexExpectWatchValue turns off
//      the floating point range evalution and defaults back to
//      pre-float evalution.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: float_range_no_arg.cpp:

int main() {
  float a = 1.0f;
  return a;  //DexLabel('check')
}

// DexExpectWatchValue('a', '1.00000', on_line=ref('check'))
