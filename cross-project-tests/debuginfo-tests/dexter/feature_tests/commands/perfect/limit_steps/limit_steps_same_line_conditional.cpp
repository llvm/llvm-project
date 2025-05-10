// Purpose:
//      Test that LimitStep commands can exist on the same from line.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: limit_steps_same_line_conditional.cpp

int main() {
  int val1 = 0;

  int placeholder;
  for(int ix = 0; ix != 4; ++ix) {
    val1 = ix;
    placeholder = ix;    // DexLabel('from')
    placeholder = ix;
    val1 += 2;           // DexLabel('to')
    placeholder = ix;    // DexLabel('extended_to')
  }
  return val1 + placeholder;
}

// DexExpectWatchValue('val1', 0, 1, 3, from_line=ref('from'), to_line=ref('extended_to'))

// DexLimitSteps('ix', 0, from_line=ref('from'), to_line=ref('to'))
// DexLimitSteps('ix', 1, from_line=ref('from'), to_line=ref('extended_to'))
