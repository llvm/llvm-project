## RISC-V has no ABI-reserved linker scratch register. If every usable GPR is
## live across a split edge, check that BOLT keeps that function unsplit rather
## than clobbering state in an AUIPC+JALR trampoline.

# RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.o %s
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
  add a0, a0, a6
  add a0, a0, a7
  add a0, a0, s2
  add a0, a0, s3
  add a0, a0, s4
  add a0, a0, s5
  add a0, a0, s6
  add a0, a0, s7
  add a0, a0, s8
  add a0, a0, s9
  add a0, a0, s10
  add a0, a0, s11
  add a0, a0, t3
  add a0, a0, t4
  add a0, a0, t5
  add a0, a0, t6
  ret
  .size _start, .-_start

  .reloc 0, R_RISCV_NONE
