// Purpose:
//      Check that \DexExpectStepOrder applies no penalty when the expected
//      order is found.
//
// REQUIRES: system-linux, lldb
//
// RUN: %dexter_base test --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" -- %s \
// RUN:     | FileCheck %s
// CHECK: expect_step_order.cpp:

int main()
{
  volatile int x = 1; // DexExpectStepOrder(1)
  volatile int y = 1; // DexExpectStepOrder(2)
  volatile int z = 1; // DexExpectStepOrder(3)
  return 0;
}
