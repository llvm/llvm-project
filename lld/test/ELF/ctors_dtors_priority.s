// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld %t -o %t.exe
// RUN: llvm-objdump -s %t.exe | FileCheck %s
// REQUIRES: x86

.globl _start
_start:
  nop

.section .ctors, "aw", @progbits
  .align 8
  .byte 1
.section .ctors.100, "aw", @progbits
  .long 2
.section .ctors.5, "aw", @progbits
  .byte 3
.section .ctors, "aw", @progbits
  .byte 4
.section .ctors, "aw", @progbits
  .byte 5

.section .dtors, "aw", @progbits
  .align 8
  .byte 0x11
.section .dtors.100, "aw", @progbits
  .long 0x12
.section .dtors.5, "aw", @progbits
  .byte 0x13
.section .dtors, "aw", @progbits
  .byte 0x14
.section .dtors, "aw", @progbits
  .byte 0x15

// CHECK:      Contents of section .ctors:
// CHECK-NEXT: 03020000 00000000 010405
// CHECK:      Contents of section .dtors:
// CHECK-NEXT: 13120000 00000000 111415
