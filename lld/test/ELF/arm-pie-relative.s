// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o --pie -o %t
// RUN: llvm-readobj -r %t | FileCheck %s
// RUN: llvm-readelf -x .got %t | FileCheck %s --check-prefix=GOT

// Test that a R_ARM_GOT_BREL relocation with PIE results in a R_ARM_RELATIVE
// dynamic relocation
 .syntax unified
 .text
 .global _start
_start:
 .word sym(GOT)

 .data
 .global sym
sym:
 .word 0

// CHECK:      Relocations [
// CHECK-NEXT:   Section (5) .rel.dyn {
// CHECK-NEXT:     0x201E4 R_ARM_RELATIVE

// GOT:      section '.got':
// GOT-NEXT: 0x000201e4 e8010300
