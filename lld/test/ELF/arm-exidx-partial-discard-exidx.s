// REQUIRES: arm
// RUN: llvm-mc --arm-add-build-attributes --triple=armv7a-linux-gnueabihf -filetype=obj %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:         /DISCARD/ : { *(.ARM.exidx.exit.text) *(.ARM.extab.exit.text)} \
// RUN:         . = 0x90000000; \
// RUN:         .ARM.exidx : { *(.ARM.exidx) } \
// RUN:         .text : { *(.text) } \
// RUN:         .exit.text : { *(.exit.text) } \
// RUN:         .rodata : { *(.rodata) } \
// RUN: } " > %t.script
// RUN: ld.lld --script %t.script %t.o -o %t
// RUN: llvm-readelf -x .ARM.exidx %t | FileCheck %s

/// The linker script /DISCARDS/ the .ARM.exidx and .ARM.extab for the
/// .exit.text . If we do not discard both sections we will end up with
/// a dangling reference. We expect no linker error for an out of range
/// relocation/dangling reference and just a single .ARM.exidx entry
/// for _start and an entry for the terminating sentinel.

// CHECK: Hex dump of section '.ARM.exidx':
// CHECK-NEXT: 0x90000000 10000000 01000000 10000000 01000000
// CHECK-NOT:  0x90000010
 .text
 .global _start
 .type _start, %function
_start:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .exit.text, "ax", %progbits
 .global exit_text
 .type exit_text, %function
exit_text:
  .fnstart
  bx lr
 .personality __gxx_personality_v0
 .handlerdata
 .long 0
 .fnend

/// Dummy definition for a reference from the personality routine created by
/// the assembler, use .data to avoid generating a cantunwind table.
 .section .rodata
 .global __aeabi_unwind_cpp_pr0
__aeabi_unwind_cpp_pr0:
 .word 0
