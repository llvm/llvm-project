## This test checks that the unreachable unconditional branch is removed
## if it is located after return instruction.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt | FileCheck %s

# CHECK: UCE removed 1 blocks

  .text
  .align 4
  .global main
  .type main, %function
main:
  b.eq 1f
  ret
  b main
1:
  mov x1, #1
  ret
  .size main, .-main
