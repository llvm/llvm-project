// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/far-arm-thumb-abs.s -o %tfar
// RUN: echo "SECTIONS { \
// RUN:          . = 0xb4; \
// RUN:          .callee1 : { *(.callee_low) } \
// RUN:          .callee2 : { *(.callee_arm_low) } \
// RUN:          .caller : { *(.text) } \
// RUN:          .callee3 : { *(.callee_high) } \
// RUN:          .callee4 : { *(.callee_arm_high) } } " > %t.script
// RUN: ld.lld --script %t.script %t %tfar -o %t2
// RUN: llvm-objdump -d --triple=armv7a-none-linux-gnueabi %t2 | FileCheck %s

/// Test BLX instruction is chosen for ARM BL/BLX instruction and Thumb callee
/// Using two callees to ensure at least one has 2-byte alignment.
 .syntax unified
 .thumb
 .section .callee_low, "ax",%progbits
 .align 2
 .type callee_low,%function
callee_low:
 bx lr
 .type callee_low2, %function
callee_low2:
 bx lr

 .section .callee_arm_low, "ax",%progbits
 .arm
 .balign 0x100
 .type callee_arm_low,%function
 .align 2
callee_arm_low:
  bx lr

.section .text, "ax",%progbits
 .arm
 .globl _start
 .balign 0x10000
 .type _start,%function
_start:
 bl  callee_low
 blx callee_low
 bl  callee_low2
 blx callee_low2
 bl  callee_high
 blx callee_high
 bl  callee_high2
 blx callee_high2
 bl  blx_far
 blx blx_far2
/// blx to ARM instruction should be written as a BL
 bl  callee_arm_low
 blx callee_arm_low
 bl  callee_arm_high
 blx callee_arm_high
 bx lr

 .section .callee_high, "ax",%progbits
 .balign 0x100
 .thumb
 .type callee_high,%function
callee_high:
 bx lr
 .type callee_high2,%function
callee_high2:
 bx lr

 .section .callee_arm_high, "ax",%progbits
 .arm
 .balign 0x100
 .type callee_arm_high,%function
callee_arm_high:
  bx lr

// CHECK: Disassembly of section .callee1:
// CHECK-EMPTY:
// CHECK-NEXT: <callee_low>:
// CHECK-NEXT:    b4:       4770    bx      lr
// CHECK: <callee_low2>:
// CHECK-NEXT:    b6:       4770    bx      lr

// CHECK: Disassembly of section .callee2:
// CHECK-EMPTY:
// CHECK-NEXT: <callee_arm_low>:
// CHECK-NEXT:    100:        e12fff1e        bx      lr

// CHECK: Disassembly of section .caller:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:   10000:       faffc02b        blx     0xb4 <callee_low>
// CHECK-NEXT:   10004:       faffc02a        blx     0xb4 <callee_low>
// CHECK-NEXT:   10008:       fbffc029        blx     0xb6 <callee_low2>
// CHECK-NEXT:   1000c:       fbffc028        blx     0xb6 <callee_low2>
// CHECK-NEXT:   10010:       fa00003a        blx     0x10100 <callee_high>
// CHECK-NEXT:   10014:       fa000039        blx     0x10100 <callee_high>
// CHECK-NEXT:   10018:       fb000038        blx     0x10102 <callee_high2>
// CHECK-NEXT:   1001c:       fb000037        blx     0x10102 <callee_high2>
/// 0x2010024 = blx_far           
// CHECK-NEXT:   10020:       fa7fffff        blx     0x2010024
/// 0x2010028 = blx_far2          
// CHECK-NEXT:   10024:       fa7fffff        blx     0x2010028
// CHECK-NEXT:   10028:       ebffc034        bl      0x100 <callee_arm_low>
// CHECK-NEXT:   1002c:       ebffc033        bl      0x100 <callee_arm_low>
// CHECK-NEXT:   10030:       eb000072        bl      0x10200 <callee_arm_high>
// CHECK-NEXT:   10034:       eb000071        bl      0x10200 <callee_arm_high>
// CHECK-NEXT:   10038:       e12fff1e        bx      lr

// CHECK: Disassembly of section .callee3:
// CHECK-EMPTY:
// CHECK: <callee_high>:
// CHECK-NEXT:    10100:       4770    bx      lr
// CHECK: <callee_high2>:
// CHECK-NEXT:    10102:       4770    bx      lr

// CHECK: Disassembly of section .callee4:
// CHECK-EMPTY:
// CHECK-NEXT: <callee_arm_high>:
// CHECK-NEXT:   10200:     e12fff1e        bx      lr
