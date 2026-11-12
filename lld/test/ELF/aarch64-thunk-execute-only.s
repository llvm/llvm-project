// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld %t.o --defsym big=0x1111222233334444 -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

.section .text,"axy",@progbits,unique,0
.globl _start
_start:
    bl big
    b  big

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-LABEL: <_start>:
// CHECK-NEXT:             bl      {{.*}} <__AArch64AbsXOLongThunk_big>
// CHECK-NEXT:             b       {{.*}} <__AArch64AbsXOLongThunk_big>
// CHECK-LABEL: <__AArch64AbsXOLongThunk_big>:
// CHECK-NEXT:             mov     x16, #0x4444
// CHECK-NEXT:             movk    x16, #0x3333, lsl #16
// CHECK-NEXT:             movk    x16, #0x2222, lsl #32
// CHECK-NEXT:             movk    x16, #0x1111, lsl #48
// CHECK-NEXT:             br      x16
