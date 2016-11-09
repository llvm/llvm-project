// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2 2>&1
// RUN: llvm-objdump -triple=thumbv7a-none-linux-gnueabi -d %t2 | FileCheck %s
// REQUIRES: arm

// Check that no thunks are created for an undefined weak symbol
 .syntax unified

.weak target

.section .text.thumb, "ax", %progbits
 .thumb
 .global
_start:
 bl target
 b target
 b.w target

// CHECK: Disassembly of section .text:
// CHECK-NEXT: _start:
// 69636 = 0x11004 = next instruction
// CHECK:         11000:        11 f0 02 f8     bl      #69636
// CHECK-NEXT:    11004:        11 f0 04 b8     b.w     #69640
// CHECK-NEXT:    11008:        11 f0 06 b8     b.w     #69644
