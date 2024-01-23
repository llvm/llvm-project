# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax c.s -o c.64.o
# RUN: ld.lld -shared -soname=c.64.so c.64.o -o c.64.so

# RUN: ld.lld -shared -z now a.64.o c.64.o -o a.64.so -z separate-code
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d a.64.so | FileCheck %s --check-prefix=GD64

# RUN: ld.lld -e 0 -z now a.64.o c.64.o -o a.64.le -z separate-code
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d a.64.le | FileCheck %s --check-prefix=LE64

# RUN: ld.lld -e 0 -z now a.64.o c.64.so -o a.64.ie -z separate-code
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d a.64.ie | FileCheck %s --check-prefix=IE64

# GD64:      .got     00000018 00000000000020c0
# GD64-LABEL: <_start>:
# GD64-NEXT:         jal     {{.*}} <foo>
# GD64-LABEL: <foo>:
## &.got[c]-. = 0x20c0+8 - 0x1004 = 0x10c4
# GD64:        1004: auipc   a2, 0x1
# GD64-NEXT:         ld      a3, 0xc4(a2)
# GD64-NEXT:         addi    a0, a2, 0xc4
# GD64-NEXT:         jalr    t0, 0x0(a3)
# GD64-NEXT:         c.add   a0, tp
# GD64-NEXT:         jal     {{.*}} <foo>
# GD64-NEXT:         auipc   a4, 0x1
# GD64-NEXT:         ld      a5, 0xae(a4)
# GD64-NEXT:         addi    a0, a4, 0xae
# GD64-NEXT:         jalr    t0, 0x0(a5)
# GD64-NEXT:         c.add   a0, tp

# LE64-LABEL: <_start>:
# LE64-NEXT:         jal     {{.*}} <foo>
# LE64-LABEL: <foo>:
# LE64-NEXT:  11004: lui     a0, 0x0
# LE64-NEXT:         addi    a0, zero, 0xc
# LE64-NEXT:         c.add   a0, tp
# LE64-NEXT:         jal     {{.*}} <foo>
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         lui     a0, 0x0
# LE64-NEXT:         addi    a0, zero, 0xc
# LE64-NEXT:         c.add   a0, tp
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         lui     a0, 0x0
# LE64-NEXT:         addi    a0, zero, 0xc
# LE64-NEXT:         c.add   a0, tp

# IE64:       .got     00000010 00000000000120e0
# IE64-LABEL: <_start>:
# IE64-NEXT:         jal     {{.*}} <foo>
# IE64-LABEL: <foo>:
## &.got[c]-. = 0x120e0+8 - 0x11004 = 0x10e4
# IE64-NEXT:  11004: auipc   a0, 0x1
# IE64-NEXT:         ld      a0, 0xe4(a0)
# IE64-NEXT:         c.add   a0, tp
# IE64-NEXT:         jal     {{.*}} <foo>
# IE64-NEXT:         addi    zero, zero, 0x0
## &.got[c]-. = 0x120e0+8 - 0x11016 = 0x10d2
# IE64-NEXT:  11016: auipc   a0, 0x1
# IE64-NEXT:         ld      a0, 0xd2(a0)
# IE64-NEXT:         c.add   a0, tp
# IE64-NEXT:         addi    zero, zero, 0x0
## &.got[c]-. = 0x120e0+8 - 0x11024 = 0x10c4
# IE64-NEXT:  11024: auipc   a0, 0x1
# IE64-NEXT:         ld      a0, 0xc4(a0)
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
  auipc a2, %tlsdesc_hi(c)
  .reloc .-4, R_RISCV_RELAX, 0
  ld    a3, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a2)
  .reloc .-4, R_RISCV_RELAX, 0
  addi  a0, a2, %tlsdesc_add_lo(.Ltlsdesc_hi0)
  .reloc .-4, R_RISCV_RELAX, 0
  jalr  t0, 0(a3), %tlsdesc_call(.Ltlsdesc_hi0)
  .reloc .-4, R_RISCV_RELAX, 0
  add   a0, a0, tp
.option relax

  call foo

.Ltlsdesc_hi1:
.option norelax
## LD has an R_RISCV_RELAX.
  auipc a4, %tlsdesc_hi(c)
  ld    a5, %tlsdesc_load_lo(.Ltlsdesc_hi1)(a4)
  .reloc .-4, R_RISCV_RELAX, 0
  addi  a0, a4, %tlsdesc_add_lo(.Ltlsdesc_hi1)
  jalr  t0, 0(a5), %tlsdesc_call(.Ltlsdesc_hi1)
  add   a0, a0, tp
.option relax

.Ltlsdesc_hi2:
.option norelax
## AUIPC has an R_RISCV_RELAX.
  auipc a6, %tlsdesc_hi(c)
  .reloc .-4, R_RISCV_RELAX, 0
  ld    a7, %tlsdesc_load_lo(.Ltlsdesc_hi2)(a6)
  addi  a0, a6, %tlsdesc_add_lo(.Ltlsdesc_hi2)
  jalr  t0, 0(a7), %tlsdesc_call(.Ltlsdesc_hi2)
  add   a0, a0, tp
.option relax

.section .tbss
.globl a
.zero 8
a:
.zero 3
b:
.zero 1

#--- c.s
.tbss
.globl c
c: .zero 4
