// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %S/Inputs/far-arm-thumb-abs.s -o %tfar
// RUN: echo "SECTIONS { \
// RUN:          . = 0xb4; \
// RUN:          .callee1 : { *(.callee_low) } \
// RUN:          .caller : { *(.text) } \
// RUN:          .callee2 : { *(.callee_high) } } " > %t.script
// RUN: ld.lld --script %t.script %t %tfar -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck  %s

 .syntax unified
 .thumb
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
 bl  far_uncond
 b   far_uncond
 bgt far_cond
 bx lr

 .section .callee_high, "ax",%progbits
 .align 2
 .type callee_high,%function
callee_high:
 bx lr

// CHECK: Disassembly of section .callee1:
// CHECK-EMPTY:
// CHECK-NEXT: <callee_low>:
// CHECK-NEXT:      b4:       4770    bx      lr
// CHECK-EMPTY:
// CHECK-NEXT: Disassembly of section .caller:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:   10000:       f7f0 f858       bl      0xb4 <callee_low>
// CHECK-NEXT:   10004:       f7f0 b856       b.w     0xb4 <callee_low>
// CHECK-NEXT:   10008:       f430 a854       beq.w   0xb4 <callee_low>
// CHECK-NEXT:   1000c:       f000 f80c       bl      0x10028 <callee_high>
// CHECK-NEXT:   10010:       f000 b80a       b.w     0x10028 <callee_high>
// CHECK-NEXT:   10014:       f040 8008       bne.w   0x10028 <callee_high>
/// far_uncond = 0x101001b
// CHECK-NEXT:   10018:       f3ff d7ff       bl      0x101001a
// CHECK-NEXT:   1001c:       f3ff 97fd       b.w     0x101001a
/// far_cond = 0x110023
// CHECK-NEXT:   10020:       f33f afff       bgt.w   0x110022
// CHECK-NEXT:   10024:       4770    bx      lr
// CHECK-NEXT:   10026:
// CHECK-EMPTY:
// CHECK-NEXT: Disassembly of section .callee2:
// CHECK-EMPTY:
// CHECK-NEXT: <callee_high>:
// CHECK-NEXT:   10028:       4770    bx      lr
