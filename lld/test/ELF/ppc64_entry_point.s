# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t
# RUN: ld.lld %t -o %t2
# RUN: llvm-objdump -d %t2 | FileCheck %s

.text
.abiversion 2
.globl  _start
.p2align        4
.type   _start,@function

_start:
.Lfunc_begin0:
.Lfunc_gep0:
  lis   4, .Lfunc_gep0@ha
  addi  4, 4, .Lfunc_gep0@l
  # now r4 should contain the address of _start

  lis   5, .TOC.-.Lfunc_gep0@ha
  addi  5, 5, .TOC.-.Lfunc_gep0@l
  # now r5 should contain the offset s.t. r4 + r5 = TOC base

  # exit 55
  li    0, 1
  li    3, 55
  sc
.Lfunc_end0:
    .size   _start, .Lfunc_end0-.Lfunc_begin0

// CHECK: 10010000:       01 10 80 3c     lis 4, 4097
// CHECK-NEXT: 10010004:       00 00 84 38     addi 4, 4, 0
// CHECK-NEXT: 10010008:       02 00 a0 3c     lis 5, 2
// CHECK-NEXT: 1001000c:       00 80 a5 38     addi 5, 5, -32768
