// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o --pie -o %t
// RUN: llvm-readobj -r %t | FileCheck %s
// RUN: llvm-readelf -x .got %t | FileCheck %s --check-prefix=GOT
// RUN: ld.lld %t.o --pie --image-base=0x80000000 --check-dynamic-relocations -o %t1 2>&1 | \
// RUN:   FileCheck %s -allow-empty -check-prefix=NOERR
// RUN: llvm-readobj -r %t1 | FileCheck %s --check-prefix=CHECK1
// RUN: llvm-readelf -x .got %t1 | FileCheck %s --check-prefix=GOT1

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

// NOERR-NOT: internal linker error

// CHECK1:      Relocations [
// CHECK1-NEXT:   Section (5) .rel.dyn {
// CHECK1-NEXT:     0x800201E4 R_ARM_RELATIVE

// GOT1:      section '.got':
// GOT1-NEXT: 0x800201e4 e8010380
