// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %S/Inputs/abs.s -o %tabs
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: not ld.lld -shared %tabs %t -o %t2 2>&1 | FileCheck %s
// REQUIRES: x86

  movl $big, %edx
  movq _start - 0x1000000000000, %rdx

# CHECK: R_X86_64_32 out of range
# CHECK: R_X86_64_32S out of range
