// Purpose:
//      Check that \DexLimitSteps works even if the opening breakpoint line
//      doesn't exist. This can happen due to optimisations or label is on an
//      empty line.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: limit_steps_line_mismatch.cpp

int main() {
  int i = 0;
  for (; i < 2; i++) {
    // DexLabel('from')
    int x = i;
  }
  int ret = 0;
  return ret; // DexLabel('to')
}

// DexLimitSteps('1', '1', from_line=ref('from'), to_line=ref('to'))
// DexExpectWatchValue('i', 0, 1, 2, from_line=ref('from'), to_line=ref('to'))
