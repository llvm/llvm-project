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
  volatile int x = 1; // DexExpectStepOrder(1)
  volatile int y = 1; // DexExpectStepOrder(2)
  volatile int z = 1; // DexExpectStepOrder(3)
  return 0;
}
