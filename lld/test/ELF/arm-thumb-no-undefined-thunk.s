// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump --triple=thumbv7a-none-linux-gnueabi -d %t2 | FileCheck %s

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
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// 0x110b8 = next instruction
// CHECK:         200b4: {{.*}} bl      #0
// CHECK-NEXT:    200b8: {{.*}} b.w     #0 <_start+0x8>
// CHECK-NEXT:    200bc: {{.*}} b.w     #0 <_start+0xc>
