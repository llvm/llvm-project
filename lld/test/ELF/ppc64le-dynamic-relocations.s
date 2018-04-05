// REQUIRES: ppc
// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/shared-ppc64le.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t.o %t2.so -o %t
// RUN: llvm-readobj -dyn-relocations %t | FileCheck %s
// RUN: llvm-objdump -D %t | FileCheck --check-prefix=DIS %s
// RUN: llvm-readelf -dynamic-table %t | FileCheck --check-prefix=DT %s

// The dynamic relocation for foo should point to 16 bytes past the start of
// the .got.plt section.
// CHECK: Dynamic Relocations {
// CHECK-NEXT:    0x10020010 R_PPC64_JMP_SLOT foo 0x0

// There should be 2 reserved doublewords before the first entry. The dynamic
// linker will fill those in with the address of the resolver entry point and
// the dynamic object identifier.
// DIS: Disassembly of section .got.plt:
// DIS-NEXT:  .got.plt:
// DIS-NEXT: 10020000:       00 00 00 00  <unknown>
// DIS-NEXT: 10020004:       00 00 00 00  <unknown>
// DIS-NEXT: 10020008:       00 00 00 00  <unknown>
// DIS-NEXT: 1002000c:       00 00 00 00  <unknown>
// DIS-NEXT: 10020010:       00 00 00 00  <unknown>
// DIS-NEXT: 10020014:       00 00 00 00  <unknown>

// DT_PLTGOT should point to the start of the .got.plt section.
// DT: 0x0000000000000003 PLTGOT               0x10020000

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
