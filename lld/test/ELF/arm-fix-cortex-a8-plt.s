// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf --arm-add-build-attributes %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:       .plt  0x2000 : { *(.plt) *(.plt.*) } \
// RUN:       .text : { *(.text) } \
// RUN:       }" > %t.script

// RUN: ld.lld --script %t.script --fix-cortex-a8 --shared -verbose %t.o -o %t2
// RUN: llvm-objdump -d --start-address=0x2020 --stop-address=0x202c --no-show-raw-insn %t2 | FileCheck --check-prefix=CHECK-PLT %s
// RUN: llvm-objdump -d --start-address=0x2ffa --stop-address=0x3008 --no-show-raw-insn %t2 | FileCheck %s

/// If we patch a branch instruction that is indirected via the PLT then we
/// must make sure the patch goes via the PLT

// CHECK-PLT:          2020:            add     r12, pc, #0, #12
// CHECK-PLT-NEXT:     2024:            add     r12, r12, #4096
// CHECK-PLT-NEXT:     2028:            ldr     pc, [r12, #68]!

 .syntax unified
 .thumb

 .global external
 .type external, %function

 .text
 .balign 2048

 .space 2042
 .global source
 .thumb_func
source:
 nop.w
 bl external

// CHECK:      00002ffa source:
// CHECK-NEXT:     2ffa:        nop.w
// CHECK-NEXT:     2ffe:        blx     #4
// CHECK:      00003004 __CortexA8657417_2FFE:
// CHECK-NEXT:     3004:        b       #-4076
