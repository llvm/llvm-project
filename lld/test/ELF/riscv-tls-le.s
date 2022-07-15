# REQUIRES: riscv

## Additionally test that (a) -no-pie/-pie have the same behavior
## (b) --no-relax/--relax have the same behavior when R_RISCV_RELAX is suppressed.
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.32.o
# RUN: ld.lld --relax %t.32.o -o %t.32
# RUN: llvm-nm -p %t.32 | FileCheck --check-prefixes=NM %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefixes=LE %s
# RUN: ld.lld -pie --no-relax %t.32.o -o %t.32
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefixes=LE %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.64.o
# RUN: ld.lld --no-relax %t.64.o -o %t.64
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefixes=LE %s
# RUN: ld.lld -pie --no-relax %t.64.o -o %t.64
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefixes=LE %s
# RUN: ld.lld %t.64.o -o %t.64.relax
# RUN: llvm-objdump -d --no-show-raw-insn %t.64.relax | FileCheck --check-prefixes=LE-RELAX %s

# RUN: not ld.lld -shared %t.32.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

# ERR: error: relocation R_RISCV_TPREL_HI20 against .LANCHOR0 cannot be used with -shared
# ERR: error: relocation R_RISCV_TPREL_LO12_I against .LANCHOR0 cannot be used with -shared
# ERR: error: relocation R_RISCV_TPREL_HI20 against a cannot be used with -shared
# ERR: error: relocation R_RISCV_TPREL_LO12_I against a cannot be used with -shared
# ERR: error: relocation R_RISCV_TPREL_HI20 against a cannot be used with -shared
# ERR: error: relocation R_RISCV_TPREL_LO12_S against a cannot be used with -shared
# ERR: error: relocation R_RISCV_TPREL_HI20 against a cannot be used with -shared
# ERR: error: relocation R_RISCV_TPREL_LO12_S against a cannot be used with -shared

# NM: {{0*}}00000008 b .LANCHOR0
# NM: {{0*}}00000800 B a

## .LANCHOR0@tprel = 8
## a@tprel = 12
# LE:      lui a1, 0
# LE-NEXT: add a1, a1, tp
# LE-NEXT: addi a1, a1, 8
# LE-NEXT: lui a2, 0
# LE-NEXT: add a2, a2, tp
# LE-NEXT: addi a2, a2, 2044
# LE-NEXT: lui a3, 0
# LE-NEXT: addi a0, a0, 1
# LE-NEXT: add a3, a3, tp
# LE-NEXT: addi a0, a0, 2
# LE-NEXT: sw a0, 2044(a3)
# LE-NEXT: lui a4, 1
# LE-NEXT: add a4, a4, tp
# LE-NEXT: sw a0, -2048(a4)
# LE-EMPTY:

# LE-RELAX:      <.text>:
# LE-RELAX-NEXT:   addi a1, tp, 8
# LE-RELAX-NEXT:   addi a2, tp, 2044
# LE-RELAX-NEXT:   addi a0, a0, 1
# LE-RELAX-NEXT:   addi a0, a0, 2
# LE-RELAX-NEXT:   sw a0, 2044(tp)
# LE-RELAX-NEXT:   lui a4, 1
# LE-RELAX-NEXT:   add a4, a4, tp
# LE-RELAX-NEXT:   sw a0, -2048(a4)
# LE-RELAX-EMPTY:

lui a1, %tprel_hi(.LANCHOR0)
add a1, a1, tp, %tprel_add(.LANCHOR0)
addi a1, a1, %tprel_lo(.LANCHOR0)

## hi20(a-4) = hi20(0x7fc) = 0. relaxable
lui a2, %tprel_hi(a-4)
add a2, a2, tp, %tprel_add(a-4)
addi a2, a2, %tprel_lo(a-4)

## hi20(a-4) = hi20(0x7fc) = 0. relaxable
## Test non-adjacent instructions.
lui a3, %tprel_hi(a-4)
addi a0, a0, 1
add a3, a3, tp, %tprel_add(a-4)
addi a0, a0, 2
sw a0, %tprel_lo(a-4)(a3)

## hi20(a) = hi20(0x800) = 1. not relaxable
lui a4, %tprel_hi(a)
add a4, a4, tp, %tprel_add(a)
sw a0, %tprel_lo(a)(a4)

.section .tbss
.space 8
.LANCHOR0:
.space 0x800-8
.globl a
a:
.zero 4
