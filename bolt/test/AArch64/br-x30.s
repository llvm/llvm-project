## Test that "br x30" is treated as a return instruction and not as an indirect
## branch.

# RUN: %clang %cflags %s -o %t.exe -Wl,-q -Wl,--entry=foo
# RUN: llvm-bolt %t.exe -o %t.bolt --print-cfg 2>&1 | FileCheck %s

# CHECK: BB Count : 2
# CHECK-NOT: UNKNOWN CONTROL FLOW

  .text
  .global foo
  .type foo, %function
foo:
  br x30
  nop
  .size foo, .-foo
