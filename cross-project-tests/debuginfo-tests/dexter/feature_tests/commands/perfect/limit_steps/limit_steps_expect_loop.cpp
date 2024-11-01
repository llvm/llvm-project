// Purpose:
//      Check the DexLimit steps only gathers step info for 2 iterations of a
//      for loop.
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: limit_steps_expect_loop.cpp:

int main(const int argc, const char * argv[]) {
  unsigned int sum = 1;
  for(unsigned int ix = 0; ix != 5; ++ix) {
    unsigned thing_to_add = ix + ix - ix;   // DexLabel('start')
    sum += ix;                              // DexLabel('end')
  }
  return sum;
}

// DexLimitSteps('ix', 0, 3, from_line=ref('start'), to_line=ref('end'))
// DexExpectWatchValue('ix', 0, 3, from_line=ref('start'), to_line=ref('end'))
