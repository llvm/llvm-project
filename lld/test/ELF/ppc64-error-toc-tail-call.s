// REQUIRES: ppc

// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: not ld.lld %t.o %t2.so -o /dev/null 2>&1 | FileCheck %s

// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: not ld.lld %t.o %t2.so -o /dev/null 2>&1 | FileCheck %s

# A tail call to an external function without a nop should issue an error.
// CHECK: call to foo lacks nop, can't restore toc
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
