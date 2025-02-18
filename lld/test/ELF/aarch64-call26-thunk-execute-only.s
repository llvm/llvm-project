// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/big-execute-only.s -o %tbig.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld %t.o %tbig.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

.section .text,"axy",@progbits,unique,0
.globl _start
_start:
    bl big

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:    210120:       bl      0x210124
// CHECK:      <__AArch64AbsXOLongThunk_big>:
// CHECK-NEXT:    210124:       mov  x16, #0x4444
// CHECK-NEXT:    210128:       movk x16, #0x3333, lsl #16
// CHECK-NEXT:    21012c:       movk x16, #0x2222, lsl #32
// CHECK-NEXT:    210130:       movk x16, #0x1111, lsl #48
// CHECK-NEXT:    210134:       br   x16

