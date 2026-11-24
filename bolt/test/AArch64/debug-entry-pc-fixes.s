## Check that BOLT rewrites DW_AT_entry_pc to the relocated entry instruction.

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux -I %S/Inputs %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --relocs --update-debug-sections

# RUN: llvm-objdump -d %t.bolt > %t.dump
# RUN: llvm-dwarfdump %t.bolt >> %t.dump
# RUN: FileCheck %s < %t.dump

# CHECK: Disassembly of section .text:
# CHECK-LABEL: <_start>:
# CHECK: [[#%x,ENTRY:]]: {{.*}} b {{.*}} <_start>

# CHECK: DW_TAG_inlined_subroutine
# CHECK-NEXT: DW_AT_abstract_origin {{.*}}"inner")
# CHECK-NEXT: DW_AT_entry_pc (0x{{0*}}[[#ENTRY]])

    .file 1 "debug-entry-pc-fixes.s"
    .type _start,@function
    .globl _start
_start:
  mov w0, #0
.Lentry:
  b _start

.include "debug-entry-pc-dwarf.inc"
