# REQUIRES: mips
# Check writing updated addend for R_MIPS_GOT16 relocation,
# when produce a relocatable output.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -o %t.o %s
# RUN: ld.lld -r -o %t %t.o %t.o
# RUN: llvm-objdump -d -r --no-show-raw-insn %t | FileCheck --check-prefix=OBJ %s
# RUN: ld.lld -shared -o %t.so %t
# RUN: llvm-objdump -d -t --print-imm-hex --no-show-raw-insn %t.so \
# RUN:   | FileCheck -check-prefix=SO %s

# OBJ:      Disassembly of section .text:
# OBJ-EMPTY:
# OBJ-NEXT: <.text>:
# OBJ-NEXT:   lw      $25, 0($gp)
# OBJ-NEXT:           00000000:  R_MIPS_GOT16 .data
# OBJ-NEXT:   addiu   $4, $25, 0
# OBJ-NEXT:           00000004:  R_MIPS_LO16  .data
# OBJ:        lw      $25, 0($gp)
# OBJ-NEXT:           00000010:  R_MIPS_GOT16 .data
# OBJ-NEXT:   addiu   $4, $25, 16
# OBJ-NEXT:           00000014:  R_MIPS_LO16  .data

# SO: SYMBOL TABLE
# SO: {{0*}}[[D1:[0-9a-f]{1,4}]] l .data {{0+}} data
# SO: {{0*}}[[D2:[0-9a-f]{1,4}]] l .data {{0+}} data

# SO:      Disassembly of section .text:
# SO-EMPTY:
# SO-NEXT: <.text>:
# SO-NEXT:    lw      $25, -0x7fe8($gp)
# SO-NEXT:    addiu   $4, $25, 0x[[D1]]
# SO:         lw      $25, -0x7fe8($gp)
# SO-NEXT:    addiu   $4, $25, 0x[[D2]]

  .text
  lw     $t9, %got(.data)($gp)
  addiu  $a0, $t9, %lo(.data)

  .data
data:
  .word 0
