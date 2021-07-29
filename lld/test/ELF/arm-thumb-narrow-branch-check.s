// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:          . = SIZEOF_HEADERS; \
// RUN:          .R_ARM_PC11_1 : { *(.R_ARM_PC11_1) } \
// RUN:          .caller : { *(.caller) } \
// RUN:          .R_ARM_PC11_2 : { *(.R_ARM_PC11_2) } \
// RUN:          .text : { *(.text) } } " > %t.script
// RUN: ld.lld --script %t.script %t %S/Inputs/arm-thumb-narrow-branch.o -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s

// Test the R_ARM_PC11 relocation which is used with the narrow encoding of B.N
// the source of these relocations is a binary file arm-thumb-narrow-branch.o
// which has been assembled with the GNU assembler as llvm-mc doesn't emit it
// as the range of +-2048 bytes is too small to be practically useful for out
// of section branches.
 .syntax unified

.global callee_low_far
.type callee_low_far,%function
callee_low_far = 0x809

 .section .R_ARM_PC11_1,"ax",%progbits
 .thumb
 .balign 0x1000
 .type callee_low,%function
 .globl callee_low
callee_low:
 bx lr

 .text
 .align 2
 .thumb
 .globl _start
 .type _start, %function
_start:
 bl callers
 bx lr

 .section .R_ARM_PC11_2,"ax",%progbits
 .thumb
 .align 2
 .type callee_high,%function
 .globl callee_high
callee_high:
 bx lr

.global callee_high_far
.type callee_high_far,%function
callee_high_far = 0x180d

// CHECK: Disassembly of section .R_ARM_PC11_1:
// CHECK-EMPTY:
// CHECK-NEXT: <callee_low>:
// CHECK-NEXT:    1000:       70 47   bx      lr
// CHECK-EMPTY:
// CHECK-NEXT: Disassembly of section .caller:
// CHECK-EMPTY:
// CHECK-NEXT: <callers>:
/// callee_low_far = 0x809
// CHECK-NEXT:    1004:       00 e4   b       0x808
// CHECK-NEXT:    1006:       fb e7   b       0x1000 <callee_low>
// CHECK-NEXT:    1008:       02 e0   b       0x1010 <callee_high>
/// callee_high_far = 0x180d
// CHECK-NEXT:    100a:       ff e3   b       0x180c
// CHECK-NEXT:    100c:       70 47   bx      lr
// CHECK-NEXT:    100e:       00 bf   nop
// CHECK-EMPTY:
// CHECK-NEXT: Disassembly of section .R_ARM_PC11_2:
// CHECK-EMPTY:
// CHECK-NEXT: <callee_high>:
// CHECK-NEXT:    1010:       70 47   bx      lr
// CHECK-EMPTY:
// CHECK-NEXT: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:    1014:       ff f7 f6 ff     bl      0x1004 <callers>
// CHECK-NEXT:    1018:       70 47   bx      lr
