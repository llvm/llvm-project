/// Test the Arm state R_ARM_CALL and R_ARM_JUMP24 relocation to Arm state destinations.
/// R_ARM_CALL is used for branch and link (BL)
/// R_ARM_JUMP24 is used for unconditional and conditional branches (B and B<cc>)
/// Relocations defined in https://github.com/ARM-software/abi-aa/blob/main/aaelf32/aaelf32.rst
/// Addend A is always -8 to cancel out Arm state PC-bias of 8 bytes

// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:          . = 0xb4; \
// RUN:          .callee1 : { *(.callee_low) } \
// RUN:          .caller : { *(.text) } \
// RUN:          .callee2 : { *(.callee_high) } } " > %t.script
// RUN: ld.lld --defsym=far=0x201001c --script %t.script %t -o %t2
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
// CHECK-NEXT:   10000:       bl      0xb4 <callee_low>
// CHECK-NEXT:   10004:       b       0xb4 <callee_low>
// CHECK-NEXT:   10008:       beq     0xb4 <callee_low>
// CHECK-NEXT:   1000c:       bl      0x10028 <callee_high>
// CHECK-NEXT:   10010:       b       0x10028 <callee_high>
// CHECK-NEXT:   10014:       bne     0x10028 <callee_high>
/// 0x201001c = far
// CHECK-NEXT:   10018:       bl      0x201001c
// CHECK-NEXT:   1001c:       b       0x201001c
// CHECK-NEXT:   10020:       bgt     0x201001c
// CHECK-NEXT:   10024:       bx      lr
