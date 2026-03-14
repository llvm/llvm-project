// REQUIRES: x86
// RUN: llvm-mc --triple=x86_64 -filetype=obj %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:         . = 0x80000000; \
// RUN:         .linkorder : { *(.linkorder.*) } \
// RUN:         .text : { *(.text) } \
// RUN:         .text.1 0x80000200 : AT(0x1000) { *(.text.1) } \
// RUN:         .text.2 0x80000100 : AT(0x2000) { *(.text.2) } \
// RUN: } " > %t.script
// RUN: ld.lld --script %t.script %t.o -o %t
// RUN: llvm-readobj -x .linkorder  %t | FileCheck %s

/// When a linker script does not have monotonically increasing addresses
/// the SHF_LINK_ORDER sections should still be in monotonically increasing
/// order.

// CHECK: Hex dump of section '.linkorder':
// CHECK-NEXT: 0x80000000 0201

.section .text.1, "ax", %progbits
.global _start
_start:
nop

.section .text.2, "ax", %progbits
.byte 0

.section .linkorder.1, "ao", %progbits, .text.1
.byte 1

.section .linkorder.2, "ao", %progbits, .text.2
.byte 2
