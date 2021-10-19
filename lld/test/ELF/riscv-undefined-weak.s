# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=RELOC %s

# RUN: ld.lld -e absolute %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefixes=CHECK,PC %s
# RUN: llvm-readelf -x .data %t | FileCheck --check-prefixes=HEX %s

# RUN: ld.lld -e absolute %t.o -o %t --export-dynamic
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefixes=CHECK,PLT %s
# RUN: llvm-readelf -x .data %t | FileCheck --check-prefixes=HEX %s

.weak target
.global absolute, relative, branch

## Absolute relocations are resolved to 0.
# RELOC:      0x0 R_RISCV_HI20 target 0x1
# RELOC-NEXT: 0x4 R_RISCV_LO12_I target 0x1

# CHECK-LABEL: <absolute>:
# CHECK-NEXT:  lui t0, 0
# CHECK-NEXT:  addi t0, t0, 1
absolute:
  lui t0, %hi(target+1)
  addi t0, t0, %lo(target+1)

## Currently, PC-relative relocations are resolved to 0.
# RELOC-NEXT: 0x8 R_RISCV_PCREL_HI20 target 0x0
# RELOC-NEXT: 0xC R_RISCV_PCREL_LO12_I .Lpcrel_hi0 0x0
# RELOC-NEXT: 0x10 R_RISCV_PCREL_HI20 target 0x2
# RELOC-NEXT: 0x14 R_RISCV_PCREL_LO12_S .Lpcrel_hi1 0x0

## 1048559 should be -0x11.
# CHECK-LABEL: <relative>:
# CHECK-NEXT:  11{{...}}: auipc a1, 1048559
# PC-NEXT:     addi a1, a1, -352
# PLT-NEXT:    addi a1, a1, -792
# CHECK-LABEL: <.Lpcrel_hi1>:
# CHECK-NEXT:  11{{...}}: auipc t1, 1048559
# PC-NEXT:     sd a2, -358(t1)
# PLT-NEXT:    sd a2, -798(t1)
relative:
  la a1, target
  sd a2, target+2, t1

## Branch relocations
## If .dynsym does not exist, an undefined weak symbol is non-preemptible.
## Treat them as PC relative relocations.
# RELOC:      0x18 R_RISCV_CALL target 0x0
# RELOC-NEXT: 0x20 R_RISCV_JAL target 0x0
# RELOC-NEXT: 0x24 R_RISCV_BRANCH target 0x0

# PC-LABEL:    <branch>:
# PC-NEXT:     auipc ra, 0
# PC-NEXT:     jalr ra
# PC-NEXT:     [[#%x,ADDR:]]:
# PC-SAME:                    j 0x[[#ADDR]]
# PC-NEXT:     [[#%x,ADDR:]]:
# PC-SAME:                    beqz zero, 0x[[#ADDR]]

## If .dynsym exists, an undefined weak symbol is preemptible.
## We create a PLT entry and redirect the reference to it.
# PLT-LABEL:   <branch>:
# PLT-NEXT:    auipc ra, 0
# PLT-NEXT:    jalr 56(ra)
# PLT-NEXT:    [[#%x,ADDR:]]:
# PLT-SAME:                   j 0x[[#ADDR]]
# PLT-NEXT:    [[#%x,ADDR:]]:
# PLT-SAME:                   beqz zero, 0x[[#ADDR]]
branch:
  call target
  jal x0, target
  beq x0, x0, target

## Absolute relocations are resolved to 0.
# RELOC:      0x0 R_RISCV_64 target 0x3
# RELOC-NEXT: 0x8 R_RISCV_32 target 0x4
# HEX:      section '.data':
# HEX-NEXT: 03000000 00000000 04000000
.data
.p2align 3
.quad target+3
.long target+4

# PC-NOT:      .plt:
# PLT:         .plt:
