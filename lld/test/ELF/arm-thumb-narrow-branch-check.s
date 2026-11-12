// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:          . = SIZEOF_HEADERS; \
// RUN:          .callee_low : { *(.callee_low) } \
// RUN:          .caller : { *(.caller) } \
// RUN:          .callee_high : { *(.callee_high) } \
// RUN:          .text : { *(.text) } } " > %t.script
// RUN: ld.lld --script %t.script %t %S/Inputs/arm-thumb-narrow-branch.o -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s

// Test the R_ARM_THM_JUMP11 and R_ARM_THM_JUMP8 relocations which are used
// with the narrow encoding of B.N and BEQ.N.
//
// The source of these relocations is a binary file arm-thumb-narrow-branch.o
// which has been assembled with the GNU assembler as llvm-mc doesn't emit it
// as the range of +-2048 bytes is too small to be practically useful for out
// of section branches.
 .syntax unified

.global callee_low_far
.type callee_low_far,%function
callee_low_far = 0x809

.global callee_low_near
.type callee_low_near,%function
callee_low_near = 0xfff

 .section .callee_low,"ax",%progbits
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

 .section .callee_high,"ax",%progbits
 .thumb
 .align 2
 .type callee_high,%function
 .globl callee_high
callee_high:
 bx lr

.global callee_high_near
.type callee_high_near,%function
callee_high_near = 0x10ff

.global callee_high_far
.type callee_high_far,%function
callee_high_far = 0x180d

// CHECK: Disassembly of section .callee_low:
// CHECK-EMPTY:
// CHECK-NEXT: <callee_low>:
// CHECK-NEXT:    1000:       4770    bx      lr
// CHECK-EMPTY:
// CHECK-NEXT: Disassembly of section .caller:
// CHECK-EMPTY:
// CHECK-NEXT: <callers>:
/// callee_low_far = 0x809
// CHECK-NEXT:    1004:       e400    b       0x808
// CHECK-NEXT:    1006:       e7fb    b       0x1000 <callee_low>
// CHECK-NEXT:    1008:       e006    b       0x1018 <callee_high>
/// callee_high_far = 0x180d
// CHECK-NEXT:    100a:       e3ff    b       0x180c
/// callee_low_near = 0xfff
// CHECK-NEXT:    100c:       d0f7    beq     0xffe
// CHECK-NEXT:    100e:       d0f7    beq     0x1000 <callee_low>
// CHECK-NEXT:    1010:       d002    beq     0x1018 <callee_high>
/// callee_high_near = 0x10ff
// CHECK-NEXT:    1012:       d074    beq     0x10fe
// CHECK-NEXT:    1014:       4770    bx      lr
// CHECK-NEXT:    1016:       46c0    mov     r8, r8
// CHECK-EMPTY:
// CHECK-NEXT: Disassembly of section .callee_high:
// CHECK-EMPTY:
// CHECK-NEXT: <callee_high>:
// CHECK-NEXT:    1018:       4770    bx      lr
// CHECK-EMPTY:
// CHECK-NEXT: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:    101c:       f7ff fff2       bl      0x1004 <callers>
// CHECK-NEXT:    1020:       4770    bx      lr
