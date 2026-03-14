// Purpose:
//      Check that \DexExpectStepOrder applies no penalty when the expected
//      order is found.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: expect_step_order.cpp:

int main() // DexLabel('main')
{
  volatile int a = 1; // DexExpectStepOrder(1)
  volatile int b = 1; // DexExpectStepOrder(2)
  volatile int c = 1; // DexExpectStepOrder(3)

  volatile int x = 1;
  volatile int y = 1;
  volatile int z = 1;
  return 0;
}

// DexExpectStepOrder(4, on_line=ref('main')+6);
// DexExpectStepOrder(5, on_line=ref('main')+7);
// DexExpectStepOrder(6, on_line=ref('main')+8);
