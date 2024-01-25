# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 --defsym PAD=0 -mattr=+c,+relax a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 --defsym PAD=5000 -mattr=+c,+relax a.s -o aa.64.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax c.s -o c.64.o
# RUN: ld.lld -shared -soname=c.64.so c.64.o -o c.64.so

# RUN: ld.lld -shared -z now a.64.o c.64.o -o a.64.so -z separate-code
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d a.64.so | FileCheck %s --check-prefix=GD64

## Test the TLSDESC to LE optimization. Also check --emit-relocs.
# RUN: ld.lld -e 0 -z now a.64.o c.64.o -o a.64.le -z separate-code --emit-relocs
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -hdr a.64.le | FileCheck %s --check-prefix=LE64
# RUN: ld.lld -e 0 -z now aa.64.o c.64.o -o aa.64.le -z separate-code
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d aa.64.le | FileCheck %s --check-prefix=LE64A

## Test the TLSDESC to IE optimization.
# RUN: ld.lld -e 0 -z now a.64.o c.64.so -o a.64.ie -z separate-code
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d a.64.ie | FileCheck %s --check-prefix=IE64

# GD64:      .got     00000018 00000000000020c0
# GD64-LABEL: <_start>:
# GD64-NEXT:         jal     {{.*}} <foo>
# GD64-LABEL: <foo>:
## &.got[c]-. = 0x20c0+8 - 0x1004 = 0x10c4
# GD64:        1004: auipc   a2, 0x1
# GD64-NEXT:         c.add   a7, a7
# GD64-NEXT:         ld      a3, 0xc4(a2)
# GD64-NEXT:         c.add   a7, a7
# GD64-NEXT:         addi    a0, a2, 0xc4
# GD64-NEXT:         c.add   a7, a7
# GD64-NEXT:         jalr    t0, 0x0(a3)
# GD64-NEXT:         c.add   a0, tp
# GD64-NEXT:         jal     {{.*}} <foo>
## &.got[c]-. = 0x20c0+8 - 0x1020 = 0x10a8
# GD64-NEXT:   1020: auipc   a4, 0x1
# GD64-NEXT:         ld      a5, 0xa8(a4)
# GD64-NEXT:         addi    a0, a4, 0xa8
# GD64-NEXT:         jalr    t0, 0x0(a5)
# GD64-NEXT:         c.add   a0, tp
## &.got[c]-. = 0x20c0+8 - 0x1032 = 0x1096
# GD64-NEXT:   1032: auipc   a6, 0x1
# GD64-NEXT:         ld      a7, 0x96(a6)
# GD64-NEXT:         addi    a0, a6, 0x96
# GD64-NEXT:         jalr    t0, 0x0(a7)
# GD64-NEXT:         c.add   a0, tp

# LE64-LABEL: <_start>:
# LE64-NEXT:         jal     {{.*}} <foo>
# LE64-LABEL: <foo>:
# LE64-NEXT:         c.add   a7, a7
# LE64-NEXT:                 R_RISCV_TLSDESC_HI20 b
# LE64-NEXT:                 R_RISCV_RELAX *ABS*
# LE64-NEXT:         c.add   a7, a7
# LE64-NEXT:                 R_RISCV_TLSDESC_LOAD_LO12 .Ltlsdesc_hi0
# LE64-NEXT:                 R_RISCV_RELAX *ABS*
# LE64-NEXT:  11008: c.add   a7, a7
# LE64-NEXT:                 R_RISCV_TLSDESC_ADD_LO12 .Ltlsdesc_hi0
# LE64-NEXT:                 R_RISCV_RELAX *ABS*
# LE64-NEXT:         addi    a0, zero, 0x7ff
# LE64-NEXT:                 R_RISCV_TLSDESC_CALL .Ltlsdesc_hi0
# LE64-NEXT:                 R_RISCV_RELAX *ABS*
# LE64-NEXT:         c.add   a0, tp
# LE64-NEXT:         jal     {{.*}} <foo>
# LE64-NEXT:                 R_RISCV_JAL foo
# LE64-NEXT:                 R_RISCV_RELAX *ABS*
# LE64-NEXT:         addi    a0, zero, 0x7ff
# LE64-NEXT:                 R_RISCV_TLSDESC_HI20 b
# LE64-NEXT:                 R_RISCV_RELAX *ABS*
# LE64-NEXT:                 R_RISCV_TLSDESC_LOAD_LO12 .Ltlsdesc_hi1
# LE64-NEXT:                 R_RISCV_TLSDESC_ADD_LO12 .Ltlsdesc_hi1
# LE64-NEXT:                 R_RISCV_TLSDESC_CALL .Ltlsdesc_hi1
# LE64-NEXT:         c.add   a0, tp
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:                 R_RISCV_TLSDESC_HI20 b
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:                 R_RISCV_TLSDESC_LOAD_LO12 .Ltlsdesc_hi2
# LE64-NEXT:                 R_RISCV_RELAX *ABS*
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:                 R_RISCV_TLSDESC_ADD_LO12 .Ltlsdesc_hi2
# LE64-NEXT:                 R_RISCV_RELAX *ABS*
# LE64-NEXT:         addi    a0, zero, 0x7ff
# LE64-NEXT:                 R_RISCV_TLSDESC_CALL .Ltlsdesc_hi2
# LE64-NEXT:         c.add   a0, tp

# LE64A-LABEL: <_start>:
# LE64A-NEXT:         jal     {{.*}} <foo>
# LE64A-LABEL: <foo>:
# LE64A-NEXT:         c.add   a7, a7
# LE64A-NEXT:         c.add   a7, a7
# LE64A-NEXT:  11008: lui     a0, 0x2
# LE64A-NEXT:         c.add   a7, a7
# LE64A-NEXT:         addi    a0, a0, -0x479
# LE64A-NEXT:         c.add   a0, tp
# LE64A-NEXT:         jal     {{.*}} <foo>
# LE64A-NEXT:         lui     a0, 0x2
# LE64A-NEXT:         addi    a0, a0, -0x479
# LE64A-NEXT:         c.add   a0, tp
# LE64A-NEXT:         addi    zero, zero, 0x0
# LE64A-NEXT:         addi    zero, zero, 0x0
# LE64A-NEXT:         lui     a0, 0x2
# LE64A-NEXT:         addi    a0, a0, -0x479
# LE64A-NEXT:         c.add   a0, tp

# IE64:       .got     00000010 00000000000120e0
# IE64-LABEL: <_start>:
# IE64-NEXT:         jal     {{.*}} <foo>
# IE64-LABEL: <foo>:
# IE64-NEXT:         c.add   a7, a7
# IE64-NEXT:         c.add   a7, a7
## &.got[c]-. = 0x120e0+8 - 0x11008 = 0x10e0
# IE64-NEXT:  11008: auipc   a0, 0x1
# IE64-NEXT:         c.add   a7, a7
# IE64-NEXT:         ld      a0, 0xe0(a0)
# IE64-NEXT:         c.add   a0, tp
# IE64-NEXT:         jal     {{.*}} <foo>
## &.got[c]-. = 0x120e0+8 - 0x11018 = 0x10d0
# IE64-NEXT:  11018: auipc   a0, 0x1
# IE64-NEXT:         ld      a0, 0xd0(a0)
# IE64-NEXT:         c.add   a0, tp
## &.got[c]-. = 0x120e0+8 - 0x1102a = 0x10be
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:  1102a: auipc   a0, 0x1
# IE64-NEXT:         ld      a0, 0xbe(a0)
# IE64-NEXT:         c.add   a0, tp

#--- a.s
.globl _start
_start:
.balign 16
  call foo

foo:
.Ltlsdesc_hi0:
.option norelax
## All 4 instructions have an R_RISCV_RELAX.
## Check that optimization/relaxation are not affected by irrelevant instructions.
  auipc a2, %tlsdesc_hi(b)
  .reloc .-4, R_RISCV_RELAX, 0
  c.add a7, a7
  ld    a3, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a2)
  .reloc .-4, R_RISCV_RELAX, 0
  c.add a7, a7
  addi  a0, a2, %tlsdesc_add_lo(.Ltlsdesc_hi0)
  .reloc .-4, R_RISCV_RELAX, 0
  c.add a7, a7
  jalr  t0, 0(a3), %tlsdesc_call(.Ltlsdesc_hi0)
  .reloc .-4, R_RISCV_RELAX, 0
  add   a0, a0, tp
.option relax

  call foo

.Ltlsdesc_hi1:
.option norelax
## AUIPC has an R_RISCV_RELAX. We perform relaxation, ignoring whether other
## instructions have R_RISCV_RELAX.
  auipc a4, %tlsdesc_hi(b)
  .reloc .-4, R_RISCV_RELAX, 0
  ld    a5, %tlsdesc_load_lo(.Ltlsdesc_hi1)(a4)
  addi  a0, a4, %tlsdesc_add_lo(.Ltlsdesc_hi1)
  jalr  t0, 0(a5), %tlsdesc_call(.Ltlsdesc_hi1)
  add   a0, a0, tp
.option relax

.Ltlsdesc_hi2:
.option norelax
## AUIPC does not have R_RISCV_RELAX. No relaxation.
  auipc a6, %tlsdesc_hi(b)
  ld    a7, %tlsdesc_load_lo(.Ltlsdesc_hi2)(a6)
  .reloc .-4, R_RISCV_RELAX, 0
  addi  a0, a6, %tlsdesc_add_lo(.Ltlsdesc_hi2)
  .reloc .-4, R_RISCV_RELAX, 0
  jalr  t0, 0(a7), %tlsdesc_call(.Ltlsdesc_hi2)
  add   a0, a0, tp
.option relax

.section .tbss
.globl a
.zero 8
a:
.zero 2039+PAD  ## Place b at 0x7ff+PAD

#--- c.s
.tbss
.globl b
b:
.zero 4
