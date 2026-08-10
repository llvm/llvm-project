# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-nm %t | FileCheck --check-prefix=NM %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-nm %t | FileCheck --check-prefix=NM %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

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

  lis   5, .TOC.-.Lfunc_gep0@ha   # R_PPC64_REL16_HA
  addi  5, 5, .TOC.-.Lfunc_gep0@l # R_PPC64_REL16_LO
  # now r5 should contain the offset s.t. r4 + r5 = TOC base

  # exit 55
  li    0, 1
  li    3, 55
  sc
.Lfunc_end0:
    .size   _start, .Lfunc_end0-.Lfunc_begin0

# NM-DAG: 00000000100281f0 d .TOC.
# NM-DAG: 00000000100101d0 T _start

# 0x100101d0 = (4097<<16) + 464
# CHECK:      100101d0:       lis 4, 4097
# CHECK-NEXT: 100101d4:       addi 4, 4, 464
# .TOC. - _start = (2<<16) - 32736
# CHECK-NEXT: 100101d8:       lis 5, 2
# CHECK-NEXT: 100101dc:       addi 5, 5, -32736
