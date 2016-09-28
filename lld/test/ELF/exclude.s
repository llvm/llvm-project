# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld -o %t1 %t
# RUN: llvm-objdump -section-headers %t1 | FileCheck %s
# CHECK-NOT:      .aaa

.globl _start
_start:
  jmp _start

.section .aaa,"ae"
 .quad 0
