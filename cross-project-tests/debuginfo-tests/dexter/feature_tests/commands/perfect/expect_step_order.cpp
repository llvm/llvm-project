// Purpose:
//      Check that \DexExpectStepOrder applies no penalty when the expected
//      order is found.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: expect_step_order.cpp:

int main()
{
  volatile int a = 1; // DexExpectStepOrder(1)
  volatile int b = 1; // DexExpectStepOrder(2)
  volatile int c = 1; // DexExpectStepOrder(3)

  volatile int x = 1;
  volatile int y = 1;
  volatile int z = 1;
  return 0;
}

// DexExpectStepOrder(4, on_line=16);
// DexExpectStepOrder(5, on_line=17);
// DexExpectStepOrder(6, on_line=18);
