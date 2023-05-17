// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv6-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:          .callee1 0x100004 : { *(.callee_low) } \
// RUN:          .caller  0x500000 : { *(.text) } \
// RUN:          .callee2 0x900004 : { *(.callee_high) } } " > %t.script
// RUN: ld.lld %t --script %t.script -o %t2
// RUN: llvm-objdump -d --triple=armv6-none-linux-gnueabi %t2 | FileCheck %s

// On older Arm Architectures such as v5 and v6 the Thumb BL and BLX relocation
// uses a slightly different encoding that has a lower range. These relocations
// are at the extreme range of what is permitted.
 .thumb
 .text
 .syntax unified
 .cpu    arm1176jzf-s
 .globl _start
 .type   _start,%function
_start:
  bl thumbfunc
  bl armfunc
  bx lr

  .section .callee_low, "ax", %progbits
  .globl thumbfunc
  .type thumbfunc, %function
thumbfunc:
  bx lr
// CHECK: Disassembly of section .callee1:
// CHECK-EMPTY:
// CHECK-NEXT: <thumbfunc>:
// CHECK-NEXT:   100004:       4770    bx      lr
// CHECK-EMPTY:
// CHECK-NEXT: Disassembly of section .caller:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:   500000:       f400 f800       bl     0x100004 <thumbfunc>
// CHECK-NEXT:   500004:       f3ff effe       blx    0x900004 <armfunc>
// CHECK-NEXT:   500008:       4770    bx      lr

  .arm
  .section .callee_high, "ax", %progbits
  .globl armfunc
  .type armfunc, %function
armfunc:
  bx lr
// CHECK: Disassembly of section .callee2:
// CHECK-EMPTY:
// CHECK-NEXT: <armfunc>:
// CHECK-NEXT:   900004:       e12fff1e        bx      lr
