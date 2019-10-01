# REQUIRES: mips
# Check reading/writing implicit addend for R_MIPS_26 relocation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.exe | FileCheck %s

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK:      __start:
# CHECK-NEXT:   20000:       jal     0x8020000

  .text
  .global __start
__start:
  jal __start+0x8000000
