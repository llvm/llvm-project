// Purpose:
//      Check that \DexExpectWatchValue float_range=0.5 considers a range
//      difference of 0.49999 to be an expected watch value for multple values.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: float_range_multiple.cpp:

int main() {
  float a = 1.0f;
  float b = 100.f;
  a = a + 0.4999f;
  a = a + b; // DexLabel('check1')
  return a;  //DexLabel('check2')
}

// DexExpectWatchValue('a', '1.0', '101.0', from_line=ref('check1'), to_line=ref('check2'), float_range=0.5)
