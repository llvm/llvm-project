# Check trap instruction encoding.

# REQUIRES: mips

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 -o %t.o %s
# RUN: ld.lld -r -o %t %t.o %t.o
# RUN: llvm-objdump --no-print-imm-hex -d -r --no-show-raw-insn %t | FileCheck --check-prefix=EB %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux -mcpu=mips32r6 -o %t.o %s
# RUN: ld.lld -r -o %t %t.o %t.o
# RUN: llvm-objdump --no-print-imm-hex -d -r --no-show-raw-insn %t | FileCheck --check-prefix=EL %s

# EB:        8:       sigrie 1
# EL:        8:       sigrie 1

  .text
  lw     $t9, %got(.data)($gp)
  addiu  $a0, $t9, %lo(.data)

  .data
data:
  .word 0
