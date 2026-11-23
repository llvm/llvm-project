// REQUIRES: ppc
// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o a.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64le %p/Inputs/shared-ppc64.s -o b.o
// RUN: ld.lld -shared b.o -o b.so
// RUN: not ld.lld a.o b.so 2>&1 | FileCheck %s --implicit-check-not=error:

// RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o a.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64 %p/Inputs/shared-ppc64.s -o b.o
// RUN: ld.lld -shared b.o -o b.so
// RUN: not ld.lld a.o b.so 2>&1 | FileCheck %s --implicit-check-not=error:

/// A tail call to an external function without a nop should issue an error.
// CHECK: error: a.o:(.text+0x0): call to foo lacks nop, can't restore toc
// CHECK-NOT: lacks nop
    .text
    .abiversion 2

.global _start
_start:
  b foo

  // gcc/gfortran 5.4, 6.3 and earlier versions do not add nop for recursive
  // calls.
  b _start
  b _start
