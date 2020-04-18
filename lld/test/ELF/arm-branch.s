// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/far-arm-abs.s -o %tfar
// RUN: echo "SECTIONS { \
// RUN:          . = 0xb4; \
// RUN:          .callee1 : { *(.callee_low) } \
// RUN:          .caller : { *(.text) } \
// RUN:          .callee2 : { *(.callee_high) } } " > %t.script
// RUN: ld.lld --script %t.script %t %tfar -o %t2
// RUN: llvm-objdump -d --no-show-raw-insn --triple=armv7a-none-linux-gnueabi %t2 | FileCheck  %s
 .syntax unified
 .section .callee_low, "ax",%progbits
 .align 2
 .type callee_low,%function
callee_low:
 bx lr

 .section .text, "ax",%progbits
 .globl _start
 .balign 0x10000
 .type _start,%function
_start:
 bl  callee_low
 b   callee_low
 beq callee_low
 bl  callee_high
 b   callee_high
 bne callee_high
 bl  far
 b   far
 bgt far
 bx lr

 .section .callee_high, "ax",%progbits
 .align 2
 .type callee_high,%function
callee_high:
 bx lr

// CHECK: 00010000 <_start>:
/// S(callee_low) = 0xb4 P = 0x10000 A = -8 = -0xff54 = -65364
// CHECK-NEXT:   10000:       bl      #-65364 <callee_low>
/// S(callee_low) = 0xb4 P = 0x10004 A = -8 = -0xff58 = -65368
// CHECK-NEXT:   10004:       b       #-65368 <callee_low>
/// S(callee_low) = 0xb4 P = 0x10008 A = -8 = -0xff5c -65372
// CHECK-NEXT:   10008:       beq     #-65372 <callee_low>
/// S(callee_high) = 0x10028 P = 0x1000c A = -8 = 0x14 = 20
// CHECK-NEXT:   1000c:       bl      #28 <callee_high>
/// S(callee_high) = 0x10028 P = 0x10010 A = -8 = 0x10 = 16
// CHECK-NEXT:   10010:       b       #24 <callee_high>
/// S(callee_high) = 0x10028 P = 0x10014 A = -8 = 0x0c =12
// CHECK-NEXT:   10014:       bne     #20 <callee_high>
/// S(far) = 0x201001c P = 0x10018 A = -8 = 0x1fffffc = 33554428
// CHECK-NEXT:   10018:       bl      #8
/// S(far) = 0x201001c P = 0x1001c A = -8 = 0x1fffff8 = 33554424
// CHECK-NEXT:   1001c:       b       #4
/// S(far) = 0x201001c P = 0x10020 A = -8 = 0x1fffff4 = 33554420
// CHECK-NEXT:   10020:       bgt     #0
// CHECK-NEXT:   10024:       bx      lr
