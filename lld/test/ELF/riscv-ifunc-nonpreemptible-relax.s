# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: ld.lld -pie --relax %t.o -o %t
# RUN: llvm-readelf -s -r %t | FileCheck %s
# CHECK: Relocation section '.rela.dyn'
# CHECK: R_RISCV_IRELATIVE {{ *}}[[IFUNC:[0-9A-Fa-f]+]]
# CHECK: Symbol table '.symtab'
# CHECK: {{0*}}[[IFUNC]] {{.*}} IFUNC {{.*}} ifunc

.text
.option relax
.globl _start

_start:
  call target
  # GOT-only reference to IFUNC (no direct relocations)
.Lgot:
  auipc t0, %got_pcrel_hi(ifunc)
  ld t1, %pcrel_lo(.Lgot)(t0)
  ret
.globl target
target:
  ret
.globl ifunc
.type ifunc, @gnu_indirect_function
ifunc:
  ret
