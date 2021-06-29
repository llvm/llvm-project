// Purpose:
//      Check that \DexExpectStepOrder correctly applies a penalty for steps
//      found out of expected order.
//
// UNSUPPORTED: system-darwin
//
// RUN: not %dexter_regression_test -- %s | FileCheck %s
// CHECK: expect_step_order.cpp:

int main()
{
    volatile int x = 1; // DexExpectStepOrder(3)
    volatile int y = 1; // DexExpectStepOrder(1)
    volatile int z = 1; // DexExpectStepOrder(2)
    return 0;
}
