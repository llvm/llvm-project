# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 %S/Inputs/abs256.s -o %t256.o
#
# RUN: ld.lld -z max-page-size=4096 %t.o %t256.o -o %t
# RUN: llvm-objdump -s --section=.data %t | FileCheck %s
#
# CHECK: Contents of section .data:
## 12158: S = 0x100, A = 0, P = 0x12158
##        S + A - P = 0xfffedfa8
## 1215c: S = 0x100, A = 1, P = 0x1215c
##        S + A - P = 0xfffedfa5
## 12160: S = 0x100, A = -1, P = 0x12160
##        S + A - P = 0xfffedf9f
# CHECK-NEXT: 12158 a8dffeff a5dffeff 9fdffeff

.globl _start
_start:
.data
  .word foo@PLT - .
  .word foo@PLT - . + 1
  .word foo@PLT - . - 1
