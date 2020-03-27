// Purpose:
//      Check that \DexExpectStepOrder correctly applies a penalty for steps
//      found out of expected order.
//
// REQUIRES: system-linux, lldb
//
// RUN: not %dexter_base test --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" -- %s \
// RUN:     | FileCheck %s
// CHECK: expect_step_order.cpp:

int main()
{
    volatile int x = 1; // DexExpectStepOrder(3)
    volatile int y = 1; // DexExpectStepOrder(1)
    volatile int z = 1; // DexExpectStepOrder(2)
    return 0;
}
