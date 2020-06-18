// REQUIRES: arm
// RUN: llvm-mc --arm-add-build-attributes --triple=armv7a-linux-gnueabihf -filetype=obj %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:         . = 0x80000000; \
// RUN:         .ARM.exidx : { *(.ARM.exidx) } \
// RUN:         .text : { *(.text) } \
// RUN:         .text.1 0x80000200 : AT(0x1000) { *(.text.1) } \
// RUN:         .text.2 0x80000100 : AT(0x2000) { *(.text.2) } \
// RUN: } " > %t.script
// RUN: ld.lld --script %t.script %t.o -o %t
// RUN: llvm-readobj -x .ARM.exidx %t | FileCheck %s

/// When a linker script does not have monotonically increasing addresses
/// the .ARM.exidx table should still be in monotonically increasing order.

// CHECK: Hex dump of section '.ARM.exidx':
// 0x80000000 + 0x28 = 0x80000028, 0x80000008 + 0xf8 = 0x80000100 
// CHECK-NEXT: 0x80000000 24000000 08849780 f8000000 20849980
// 0x80000010 + 0x1f0 = 0x8000200, 0x80000018 + 0x1ec = 0x8000204
// CHECK-NEXT: 0x80000010 f0010000 10849880 ec010000 01000000

 .text
 .global _start
 .type _start, %function
_start:
 .fnstart
 bx lr
 .save {r7, lr}
 .setfp r7, sp, #0
 .fnend

 .section .text.1, "ax", %progbits
 .global fn1
 .type fn1, %function
fn1:
 .fnstart
 bx lr
 .save {r8, lr}
 .setfp r8, sp, #0
 .fnend

 .section .text.2, "ax", %progbits
 .global fn2
 .type fn2, %function
fn2:
 .fnstart
 bx lr
 .save {r9, lr}
 .setfp r9, sp, #0
 .fnend

/// Dummy definition for a reference from the personality routine created by
/// the assembler, use .data to avoid generating a cantunwind table.
 .section .rodata
 .global __aeabi_unwind_cpp_pr0
__aeabi_unwind_cpp_pr0:
 .word 0
