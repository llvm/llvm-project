# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.32.o
# RUN: ld.lld -pie %t.32.o -o %t.32
# RUN: llvm-readobj -r %t.32 | FileCheck --check-prefix=RELOC32 %s
# RUN: llvm-readelf -s %t.32 | FileCheck --check-prefix=SYM32 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefix=DIS32 %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.64.o
# RUN: ld.lld -pie %t.64.o -o %t.64
# RUN: llvm-readobj -r %t.64 | FileCheck --check-prefix=RELOC64 %s
# RUN: llvm-readelf -s %t.64 | FileCheck --check-prefix=SYM64 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefix=DIS64 %s

# RELOC32:      .rela.dyn {
# RELOC32-NEXT:   0x3218 R_RISCV_IRELATIVE - 0x117C
# RELOC32-NEXT: }

# SYM32: 0001190 0 FUNC GLOBAL DEFAULT {{.*}} func

# DIS32:      <_start>:
# DIS32-NEXT: 1180: auipc a0, 0
# DIS32-NEXT:       addi a0, a0, 16
# DIS32:      Disassembly of section .iplt:
# DIS32:      <func>:
## 32-bit: &.got.plt[func]-. = 0x3218-0x1190 = 4096*2+136
# DIS32-NEXT: 1190: auipc t3, 2
# DIS32-NEXT:       lw t3, 136(t3)
# DIS32-NEXT:       jalr t1, t3
# DIS32-NEXT:       nop

# RELOC64:      .rela.dyn {
# RELOC64-NEXT:   0x3370 R_RISCV_IRELATIVE - 0x1260
# RELOC64-NEXT: }

# SYM64: 000000000001270 0 FUNC GLOBAL DEFAULT {{.*}} func

# DIS64:      <_start>:
# DIS64-NEXT: 1264: auipc a0, 0
# DIS64-NEXT:       addi a0, a0, 12
# DIS64:      Disassembly of section .iplt:
# DIS64:      <func>:
## 64-bit: &.got.plt[func]-. = 0x3370-0x1270 = 4096*2+256
# DIS64-NEXT: 1270: auipc t3, 2
# DIS64-NEXT:       ld t3, 256(t3)
# DIS64-NEXT:       jalr t1, t3
# DIS64-NEXT:       nop

.text
.globl func
.type func, @gnu_indirect_function
func:
  ret

.globl _start
_start:
.L:
  auipc a0, %pcrel_hi(func)
  addi a0, a0, %pcrel_lo(.L)
