// Purpose:
//      Check that \DexExpectWatchValue float_range=0.0 matches exact values.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: float_range_zero_match.cpp:

int main() {
  float a = 1.0f;
  return a; //DexLabel('check')
}

// DexExpectWatchValue('a', '1.0000000', on_line=ref('check'), float_range=0.0)
