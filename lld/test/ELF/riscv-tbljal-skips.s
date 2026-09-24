# REQUIRES: riscv

## Coverage tests for bunch of checks in the source code.

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+zcmt %s -o %t.o
# RUN: ld.lld %t.o --relax-tbljal --defsym far=0x200000 -o %t
# RUN: llvm-objdump -d --mattr=+zcmt --no-show-raw-insn %t | FileCheck %s

## Only the relaxable calls (with R_RISCV_RELAX) should become cm.jalt/cm.jt.
# CHECK-COUNT-20: cm.jt
# CHECK-NOT:      cm.jt
# CHECK-NOT:      cm.jalt

.global _start
.p2align 3
_start:
  ## 20 relaxable tails to "far" -- enough savings to be profitable.
  .rept 20
  tail far
  .endr

  ## A tail wrapped in .option norelax should not become "zcmt calls"
.option push
.option norelax
  tail far
.option pop

  ## An auipc+jalr pair with rd=t0, which is neither x0 nor ra, should
  ## also be skipped, even if they are marked as relaxable
.option push
.Ltmp:
  auipc t0, 0
  jalr  t0, t0, 0
.option relax
  .reloc .Ltmp, R_RISCV_CALL_PLT, far
  .reloc .Ltmp, R_RISCV_RELAX, 0
.option pop
