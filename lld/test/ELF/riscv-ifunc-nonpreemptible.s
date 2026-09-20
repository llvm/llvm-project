# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -mattr=+relax -o %t.32.o
# DEFINE: %{layout} = --section-start .rela.dyn=0x1000 -Ttext=0x2000 --section-start=.iplt=0x3000
# RUN: ld.lld -pie %{layout} %t.32.o -o %t.32
# RUN: ld.lld -pie %{layout} %t.32.o -o %t.32-apply --apply-dynamic-relocs
# RUN: llvm-readobj -r -x .got.plt %t.32 | FileCheck --check-prefixes=RELOC32,NO-APPLY-RELOC32 %s
# RUN: llvm-readobj -r -x .got.plt %t.32-apply | FileCheck --check-prefixes=RELOC32,APPLY-RELOC32 %s
# RUN: llvm-readelf -s %t.32 | FileCheck --check-prefix=SYM32 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefix=DIS32 %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -mattr=+relax -o %t.64.o
# RUN: ld.lld -pie %{layout} %t.64.o -o %t.64
# RUN: ld.lld -pie %{layout} %t.64.o -o %t.64-apply --apply-dynamic-relocs
# RUN: llvm-readobj -r -x .got.plt %t.64 | FileCheck --check-prefixes=RELOC64,NO-APPLY-RELOC64 %s
# RUN: llvm-readobj -r -x .got.plt %t.64-apply | FileCheck --check-prefixes=RELOC64,APPLY-RELOC64 %s
# RUN: llvm-readelf -s %t.64 | FileCheck --check-prefix=SYM64 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefix=DIS64 %s

## ifunc0 has a direct relocation, so it gets canonicalized to the IPLT entry.
## ifunc1 has only a GOT relocation, so its symbol remains in the original section.
## ifunc2 has both direct and GOT relocations, so it gets canonicalized to the IPLT entry.
## All IRELATIVE addends must be correctly adjusted after relaxation.

# RELOC32:      .rela.dyn {
# RELOC32-NEXT:   0x50D8 R_RISCV_RELATIVE - 0x3020
# RELOC32-NEXT:   0x60DC R_RISCV_IRELATIVE - 0x2028
# RELOC32-NEXT:   0x60E0 R_RISCV_IRELATIVE - 0x202C
# RELOC32-NEXT:   0x60E4 R_RISCV_IRELATIVE - 0x2030
# RELOC32-NEXT: }
# RELOC32-LABEL:    Hex dump of section '.got.plt':
# NO-APPLY-RELOC32: 0x000060dc 00000000 00000000 00000000
# APPLY-RELOC32:    0x000060dc 28200000 2c200000 30200000

# SYM32:      {{0*}}3000 0 FUNC  GLOBAL DEFAULT {{.*}} ifunc0
# SYM32-NEXT: {{0*}}202c 0 IFUNC GLOBAL DEFAULT {{.*}} ifunc1
# SYM32-NEXT: {{0*}}3020 0 FUNC  GLOBAL DEFAULT {{.*}} ifunc2

# DIS32:      <_start>:
# DIS32-NEXT:   2000: jal 0x2024 <func>
# DIS32:      <.L0>:
# DIS32-NEXT:   2004: auipc a0, 0x1
# DIS32-NEXT:         addi a0, a0, -0x4
# DIS32:      <.L1>:
# DIS32-NEXT:   200c: auipc a1, 0x4
# DIS32-NEXT:         addi a1, a1, 0xd4
# DIS32:      <.L2>:
# DIS32-NEXT:   2014: auipc a2, 0x1
# DIS32-NEXT:         addi a2, a2, 0xc
# DIS32:      <.L3>:
# DIS32-NEXT:   201c: auipc a3, 0x3
# DIS32-NEXT:         addi a3, a3, 0xbc
# DIS32:      Disassembly of section .iplt:
# DIS32:      <ifunc0>:
## 32-bit: &.got.plt[ifunc0]-. = 0x60dc-0x3000 = 4096*3+0xdc
# DIS32-NEXT:   3000: auipc t3, 0x3
# DIS32-NEXT:         lw t3, 0xdc(t3)

# RELOC64:      .rela.dyn {
# RELOC64-NEXT:   0x5150 R_RISCV_RELATIVE - 0x3020
# RELOC64-NEXT:   0x6158 R_RISCV_IRELATIVE - 0x2028
# RELOC64-NEXT:   0x6160 R_RISCV_IRELATIVE - 0x202C
# RELOC64-NEXT:   0x6168 R_RISCV_IRELATIVE - 0x2030
# RELOC64-NEXT: }
# RELOC64-LABEL:    Hex dump of section '.got.plt':
# NO-APPLY-RELOC64: 0x00006158 00000000 00000000 00000000 00000000
# APPLY-RELOC64:    0x00006158 28200000 00000000 2c200000 00000000

# SYM64:      {{0*}}3000 0 FUNC  GLOBAL DEFAULT {{.*}} ifunc0
# SYM64-NEXT: {{0*}}202c 0 IFUNC GLOBAL DEFAULT {{.*}} ifunc1
# SYM64-NEXT: {{0*}}3020 0 FUNC  GLOBAL DEFAULT {{.*}} ifunc2

# DIS64:      <_start>:
# DIS64-NEXT:   2000: jal 0x2024 <func>
# DIS64:      <.L0>:
# DIS64-NEXT:   2004: auipc a0, 0x1
# DIS64-NEXT:         addi a0, a0, -0x4
# DIS64:      <.L1>:
# DIS64-NEXT:   200c: auipc a1, 0x4
# DIS64-NEXT:         addi a1, a1, 0x154
# DIS64:      <.L2>:
# DIS64-NEXT:   2014: auipc a2, 0x1
# DIS64-NEXT:         addi a2, a2, 0xc
# DIS64:      <.L3>:
# DIS64-NEXT:   201c: auipc a3, 0x3
# DIS64-NEXT:         addi a3, a3, 0x134
# DIS64:      Disassembly of section .iplt:
# DIS64:      <ifunc0>:
## 64-bit: &.got.plt[ifunc0]-. = 0x6158-0x3000 = 4096*3+0x158
# DIS64-NEXT:   3000: auipc t3, 0x3
# DIS64-NEXT:         ld t3, 0x158(t3)

.text
.globl _start
_start:
  call func
.L0:
  auipc a0, %pcrel_hi(ifunc0)
  addi a0, a0, %pcrel_lo(.L0)
.L1:
  auipc a1, %got_pcrel_hi(ifunc1)
  addi a1, a1, %pcrel_lo(.L1)
.L2:
  auipc a2, %pcrel_hi(ifunc2)
  addi a2, a2, %pcrel_lo(.L2)
.L3:
  auipc a3, %got_pcrel_hi(ifunc2)
  addi a3, a3, %pcrel_lo(.L3)

.globl func
func:
  ret

## Resolvers are after relaxed code, so their addresses shift due to relaxation.
## The IRELATIVE addends must be adjusted accordingly.
.globl ifunc0, ifunc1, ifunc2
.type ifunc0, @gnu_indirect_function
.type ifunc1, @gnu_indirect_function
.type ifunc2, @gnu_indirect_function
ifunc0:
  ret
ifunc1:
  ret
ifunc2:
  ret
