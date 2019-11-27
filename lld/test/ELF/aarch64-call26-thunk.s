// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs.s -o %tabs.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld %t.o %tabs.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

.text
.globl _start
_start:
    bl big

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: _start:
// CHECK-NEXT:    210120:       bl      #8
// CHECK: __AArch64AbsLongThunk_big:
// CHECK-NEXT:    210128:       ldr     x16, #8
// CHECK-NEXT:    21012c:       br      x16
// CHECK: $d:
// CHECK-NEXT:    210130:       00 00 00 00     .word   0x00000000
// CHECK-NEXT:    210134:       10 00 00 00     .word   0x00000010

