## RVE only has x0-x15. If every usable RVE GPR is live across a split edge,
## check that BOLT does not select an unavailable x16-x31 register for the
## AUIPC+JALR trampoline and keeps the function unsplit instead.

# RUN: llvm-mc -triple riscv32 -mattr=+e -filetype=obj -o %t.o %s
# RUN: ld.lld --emit-relocs -e _start -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt -split-functions \
# RUN:   -split-strategy=random2 -bolt-seed=1 2>&1 | FileCheck %s
# RUN: llvm-readelf -S %t.bolt | FileCheck --check-prefix=SECTIONS %s

# CHECK: BOLT-WARNING: keeping _start unsplit: no dead register for a RISC-V long jump
# SECTIONS-NOT: .text.cold

  .text
  .globl _start
  .type _start, @function
_start:
  beq a0, zero, .Lcold
  ret
.Lcold:
  add a0, a0, t0
  add a0, a0, t1
  add a0, a0, t2
  add a0, a0, s1
  add a0, a0, a1
  add a0, a0, a2
  add a0, a0, a3
  add a0, a0, a4
  add a0, a0, a5
  ret
  .size _start, .-_start

  .reloc 0, R_RISCV_NONE
