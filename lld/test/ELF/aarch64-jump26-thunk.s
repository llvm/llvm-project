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
// CHECK-NEXT: <_start>:
// CHECK-NEXT:    210120:       b       0x210124
// CHECK: <__AArch64AbsLongThunk_big>:
// CHECK-NEXT:    210124:       ldr     x16, 0x21012c
// CHECK-NEXT:    210128:       br      x16
// CHECK: <$d>:
// CHECK-NEXT:    21012c:       00 00 00 00     .word   0x00000000
// CHECK-NEXT:    210130:       10 00 00 00     .word   0x00000010
