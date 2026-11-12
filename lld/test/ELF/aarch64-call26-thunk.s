// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs.s -o %tabs.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld %t.o %tabs.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

.text
.globl _start
_start:
    bl big

// CHECK-LABEL: <_start>:
// CHECK-NEXT: 210120: bl      0x210128 <__AArch64AbsLongThunk_big>
// CHECK-NEXT:         udf     #0x0
// CHECK-EMPTY:
// CHECK-LABEL: <__AArch64AbsLongThunk_big>:
// CHECK-NEXT: 210128: ldr     x16, 0x210130 <__AArch64AbsLongThunk_big+0x8>
// CHECK-NEXT:         br      x16
// CHECK-NEXT: 00 00 00 00   .word   0x00000000
// CHECK-NEXT: 10 00 00 00   .word   0x00000010

