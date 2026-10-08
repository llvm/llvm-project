# REQUIRES: riscv
## R_RISCV_RVC_BRANCH and R_RISCV_RVC_JUMP encode 9-bit and 12-bit signed
## PC-relative offsets. RVC_BRANCH applies to c.beqz/c.bnez; RVC_JUMP applies
## to c.j and (rv32-only) c.jal — different opcode bits, same imm layout.
## Use .reloc to attach the compressed relocation; without it the assembler
## converts a cross-section c.beqz/c.j to an extended branch sequence.

# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c %s -o a.o

## c.beqz val=-32  = 0b111100000     : imm8, imm7_6, imm5
## c.bnez val=+30  = 0b000011110     : imm4_3, imm2_1
## c.j    val=-128 = 0b111110000000  : imm11, imm10, imm9_8, imm7
## c.jal  val=+126 = 0b000001111110  : imm6, imm5, imm4, imm3_1 + opcode mask
# RUN: ld.lld -Ttext=0x10000 a.o --defsym ba=_start-32 --defsym bb=_start+32 \
# RUN:   --defsym ja=_start-124 --defsym jb=_start+132
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn a.out | FileCheck %s
# CHECK-LABEL: <_start>:
# CHECK-NEXT:    c.beqz a0, 0xffe0
# CHECK-NEXT:    c.bnez a0, 0x10020
# CHECK-NEXT:    c.j 0xff84
# CHECK-NEXT:    c.jal 0x10084

## Out of range. Untouched relocations stay in range.
# RUN: not ld.lld -Ttext=0x10000 a.o --defsym ba=_start+0x100 --defsym bb=_start \
# RUN:   --defsym ja=_start+0x804 --defsym jb=_start+6 2>&1 \
# RUN:   | FileCheck --check-prefix=RANGE %s --implicit-check-not=error:
# RANGE: error: a.o:(.text+0x0): relocation R_RISCV_RVC_BRANCH out of range: 256 is not in [-256, 255]; references 'ba'
# RANGE: error: a.o:(.text+0x4): relocation R_RISCV_RVC_JUMP out of range: 2048 is not in [-2048, 2047]; references 'ja'

## Misalignment.
# RUN: not ld.lld -Ttext=0x10000 a.o --defsym ba=_start+5 --defsym bb=_start \
# RUN:   --defsym ja=_start+5 --defsym jb=_start+6 2>&1 \
# RUN:   | FileCheck --check-prefix=ALIGN %s --implicit-check-not=error:
# ALIGN: error: a.o:(.text+0x0): improper alignment for relocation R_RISCV_RVC_BRANCH: 0x5 is not aligned to 2 bytes
# ALIGN: error: a.o:(.text+0x4): improper alignment for relocation R_RISCV_RVC_JUMP: 0x1 is not aligned to 2 bytes

.option arch, +c
.global _start
_start:
  .reloc ., R_RISCV_RVC_BRANCH, ba
  c.beqz a0, 0
  .reloc ., R_RISCV_RVC_BRANCH, bb
  c.bnez a0, 0
  .reloc ., R_RISCV_RVC_JUMP, ja
  c.j 0
  .reloc ., R_RISCV_RVC_JUMP, jb
  c.jal 0
