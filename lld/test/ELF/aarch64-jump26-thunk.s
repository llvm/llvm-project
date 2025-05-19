// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs.s -o %tabs.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld %t.o %tabs.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

.text
.globl _start
_start:
    b big

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-LABEL: <_start>:
// CHECK-NEXT: 210120: b       0x210128
// CHECK-NEXT:         udf      #0x0
// CHECK-LABEL: <__AArch64AbsLongThunk_big>:
// CHECK-NEXT: 210128: ldr     x16, 0x210130
// CHECK-NEXT:         br      x16
// CHECK-NEXT: 00 00 00 00     .word   0x00000000
// CHECK-NEXT: 10 00 00 00     .word   0x00000010
