// REQUIRES: ppc
// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/shared-ppc64le.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t.o %t2.so -o %t
// RUN: llvm-objdump -d %t | FileCheck %s

// CHECK:          Disassembly of section .text:
// CHECK:            _start:
// CHECK:            bl .+24
        .text
        .abiversion 2
        .globl  _start
        .p2align        4
        .type   _start,@function
_start:
.Lfunc_begin0:
.Lfunc_gep0:
  addis 2, 12, .TOC.-.Lfunc_gep0@ha
  addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
  .localentry     _start, .Lfunc_lep0-.Lfunc_gep0
  bl foo
  nop
  li 0, 1
  sc
  .size _start, .-.Lfunc_begin0



// CHECK:          Disassembly of section .plt:
// CHECK:          .plt:
// CHECK-NEXT:            18 00 41 f8     std 2, 24(1)
// CHECK-NEXT:            fe ff 82 3d     addis 12, 2, -2
// CHECK-NEXT:            40 7f 8c e9     ld 12, 32576(12)
// CHECK-NEXT:            a6 03 89 7d     mtctr 12
// CHECK:                 20 04 80 4e     bctr
