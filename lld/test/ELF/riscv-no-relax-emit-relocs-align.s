# REQUIRES: riscv
## Test that --emit-relocs plus --no-relax produces correct offsets
## when R_RISCV_ALIGN is in effect.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.64.o
# RUN: ld.lld -Ttext=0x10000 --emit-relocs --no-relax %t.32.o -o %t.32
# RUN: ld.lld -Ttext=0x10000 --emit-relocs --no-relax %t.64.o -o %t.64
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases %t.32 | FileCheck %s --check-prefixes=CHECK32
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases %t.64 | FileCheck %s --check-prefixes=CHECK64

.text
.globl _start
_start:
  .reloc ., R_RISCV_ALIGN, 0x4
  nop
.L1:
  auipc a0, %pcrel_hi(_start)
  addi a0, a0, %pcrel_lo(.L1)
  nop
## For catching any trailing relocs
  nop
.size _start, .-_start

# CHECK32:      0000000000010000 <_start>:
# CHECK32-NEXT:     auipc a0, 0x0
# CHECK32-NEXT:         R_RISCV_ALIGN *ABS*+0x4
# CHECK32-NEXT:         R_RISCV_PCREL_HI20 _start
# CHECK32-NEXT:         R_RISCV_RELAX *ABS*
# CHECK32-NEXT:     addi a0, a0, 0x0
# CHECK32-NEXT:         R_RISCV_PCREL_LO12_I .L1
# CHECK32-NEXT:         R_RISCV_RELAX *ABS*
# CHECK32-NEXT:     addi zero, zero, 0x0
# CHECK32-NEXT:     addi zero, zero, 0x0

# CHECK64:      0000000000010000 <_start>:
# CHECK64-NEXT:     auipc a0, 0x0
# CHECK64-NEXT:         R_RISCV_ALIGN *ABS*+0x4
# CHECK64-NEXT:         R_RISCV_PCREL_HI20 _start
# CHECK64-NEXT:         R_RISCV_RELAX *ABS*
# CHECK64-NEXT:     addi a0, a0, 0x0
# CHECK64-NEXT:         R_RISCV_PCREL_LO12_I .L1
# CHECK64-NEXT:         R_RISCV_RELAX *ABS*
# CHECK64-NEXT:     addi zero, zero, 0x0
# CHECK64-NEXT:     addi zero, zero, 0x0
