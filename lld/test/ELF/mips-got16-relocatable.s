# REQUIRES: mips
# Check writing updated addend for R_MIPS_GOT16 relocation,
# when produce a relocatable output.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -o %t.o %s
# RUN: ld.lld -r -o %t %t.o %t.o
# RUN: llvm-objdump -d -r --no-show-raw-insn %t | FileCheck -check-prefix=OBJ %s
# RUN: ld.lld -shared -o %t.so %t
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck -check-prefix=SO %s

# OBJ:      Disassembly of section .text:
# OBJ-EMPTY:
# OBJ-NEXT: .text:
# OBJ-NEXT:        0:       lw      $25, 0($gp)
# OBJ-NEXT:                         00000000:  R_MIPS_GOT16 .data
# OBJ-NEXT:        4:       addiu   $4, $25, 0
# OBJ-NEXT:                         00000004:  R_MIPS_LO16  .data
# OBJ:            10:       lw      $25, 0($gp)
# OBJ-NEXT:                         00000010:  R_MIPS_GOT16 .data
# OBJ-NEXT:       14:       addiu   $4, $25, 16
# OBJ-NEXT:                         00000014:  R_MIPS_LO16  .data

# SO:      Disassembly of section .text:
# SO-EMPTY:
# SO-NEXT: .text:
# SO-NEXT:    10000:       lw      $25, -32744($gp)
# SO-NEXT:    10004:       addiu   $4, $25, 0
# SO:         10010:       lw      $25, -32744($gp)
# SO-NEXT:    10014:       addiu   $4, $25, 16

  .text
  lw     $t9, %got(.data)($gp)
  addiu  $a0, $t9, %lo(.data)

  .data
data:
  .word 0
