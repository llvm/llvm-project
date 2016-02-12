// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
// RUN:   %p/Inputs/ctors_dtors_priority1.s -o %t-crtbegin.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
// RUN:   %p/Inputs/ctors_dtors_priority2.s -o %t2
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
// RUN:   %p/Inputs/ctors_dtors_priority3.s -o %t-crtend.o
// RUN: ld.lld %t1 %t2 %t-crtend.o %t-crtbegin.o -o %t.exe
// RUN: llvm-objdump -s %t.exe | FileCheck %s
// REQUIRES: x86

.globl _start
_start:
  nop

.section .ctors, "aw", @progbits
  .byte 1
.section .ctors.100, "aw", @progbits
  .byte 2
.section .ctors.005, "aw", @progbits
  .byte 3
.section .ctors, "aw", @progbits
  .byte 4
.section .ctors, "aw", @progbits
  .byte 5

.section .dtors, "aw", @progbits
  .byte 0x11
.section .dtors.100, "aw", @progbits
  .byte 0x12
.section .dtors.005, "aw", @progbits
  .byte 0x13
.section .dtors, "aw", @progbits
  .byte 0x14
.section .dtors, "aw", @progbits
  .byte 0x15

// CHECK:      Contents of section .ctors:
// CHECK-NEXT: a1010405 b10302c1
// CHECK:      Contents of section .dtors:
// CHECK-NEXT: a2111415 b21312c2
