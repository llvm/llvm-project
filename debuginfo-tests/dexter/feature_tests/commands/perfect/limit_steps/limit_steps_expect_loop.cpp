// Purpose:
//      Check the DexLimit steps only gathers step info for 2 iterations of a
//      for loop.
//
// REQUIRES: system-linux
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: limit_steps_expect_loop.cpp:

int main(const int argc, const char * argv[]) {
  unsigned int sum = 1;
  for(unsigned int ix = 0; ix != 5; ++ix) {
    unsigned thing_to_add = ix + ix - ix;   // DexLabel('start')
    sum += ix;                              // DexLabel('end')
  }
  return sum;
}

// DexLimitSteps('ix', 0, 3, from_line='start', to_line='end')
// DexExpectWatchValue('ix', 0, 3, from_line='start', to_line='end')
