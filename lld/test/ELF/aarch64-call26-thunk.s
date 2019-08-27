// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %S/Inputs/abs.s -o %tabs
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %s -o %t
// RUN: ld.lld %t %tabs -o %t2 2>&1
// RUN: llvm-objdump -d -triple=aarch64-pc-freebsd %t2 | FileCheck %s

.text
.globl _start
_start:
    bl big

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: _start:
// CHECK-NEXT:    210120:        02 00 00 94     bl      #8
// CHECK: __AArch64AbsLongThunk_big:
// CHECK-NEXT:    210128:        50 00 00 58     ldr     x16, #8
// CHECK-NEXT:    21012c:        00 02 1f d6     br      x16
// CHECK: $d:
// CHECK-NEXT:    210130:        00 00 00 00     .word   0x00000000
// CHECK-NEXT:    210134:        10 00 00 00     .word   0x00000010

