# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax a.s -o rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax a.s -o rv64.o
# RUN: ld.lld --relax-gp rv32.o lds -o rv32
# RUN: ld.lld --relax-gp rv64.o lds -o rv64
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases rv32 | FileCheck %s --check-prefixes=GP,ZERO,KEEP
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases rv64 | FileCheck %s --check-prefixes=GP,ZERO,KEEP
# RUN: ld.lld --no-relax-gp rv64.o lds -o no-gp
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases no-gp | FileCheck %s --check-prefixes=NO-GP,ZERO,KEEP
# RUN: ld.lld --relax-gp --no-relax rv64.o lds -o no-relax
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases no-relax | FileCheck %s --check-prefixes=NO-GP,NO-ZERO,KEEP
# RUN: ld.lld --relax-gp -pie rv64.o lds -o pie
# RUN: ld.lld --relax-gp -shared rv64.o lds -o shared
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases pie | FileCheck %s --check-prefixes=NO-GP,NO-ZERO,KEEP
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases shared | FileCheck %s --check-prefixes=NO-GP,NO-ZERO,KEEP
# RUN: ld.lld -r rv64.o -o relocatable
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases relocatable | FileCheck %s --check-prefixes=NO-GP,NO-ZERO,KEEP
# RUN: ld.lld --relax-gp --emit-relocs rv64.o lds -o emit-relocs
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases emit-relocs | FileCheck %s --check-prefixes=GP,ZERO,KEEP
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax no-gp.s -o no-gp.o
# RUN: ld.lld --relax-gp -Ttext=0x10000 --section-start=.sdata=0x200000 no-gp.o -o missing-gp
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases missing-gp | FileCheck %s --check-prefix=NO-GP

# GP:      <_start>:
# GP-NEXT: addi a0, gp, -0x800
# NO-GP:      <_start>:
# NO-GP-NEXT: auipc a0, {{.*}}
# NO-GP-NEXT: addi a0, a0, {{.*}}

# ZERO:      <.Lpcrel_hi1>:
# ZERO-NEXT: addi a1, zero, 0x400
# NO-ZERO:      <.Lpcrel_hi1>:
# NO-ZERO-NEXT: auipc a1, {{.*}}
# NO-ZERO-NEXT: addi a1, a1, {{.*}}

# KEEP:      <init_gp>:
# KEEP-NEXT: auipc gp, {{.*}}
# KEEP-NEXT: addi gp, gp, {{.*}}
# KEEP:      <mixed>:
# KEEP-NEXT: auipc a0, {{.*}}
# KEEP-NEXT: addi a1, a0, {{.*}}
# KEEP-NEXT: lw a2, {{.*}}(a0)
# KEEP:      <unhinted_high>:
# KEEP-NEXT: auipc a0, {{.*}}
# KEEP-NEXT: addi a1, a0, {{.*}}
# KEEP:      <unused_high>:
# KEEP-NEXT: auipc a0, {{.*}}
# KEEP:      <got>:
# KEEP-NEXT: auipc a3, {{.*}}
# KEEP-NEXT: {{l[wd]}} a3, {{.*}}(a3)

#--- a.s
.globl _start
_start:
  lla a0, foo
  lla a1, low

init_gp:
  .option push
  .option norelax
  lla gp, __global_pointer$
  .option pop

mixed:
  auipc a0, %pcrel_hi(foo)
  addi a1, a0, %pcrel_lo(mixed)
  .option push
  .option norelax
  lw a2, %pcrel_lo(mixed)(a0)
  .option pop

unhinted_high:
  .option push
  .option norelax
  auipc a0, %pcrel_hi(foo)
  .option pop
  addi a1, a0, %pcrel_lo(unhinted_high)

unused_high:
  auipc a0, %pcrel_hi(foo)

got:
  .option push
  .option pic
  la a3, foo
  .option pop

.section .sdata,"aw"
foo:
  .word 0
  .space 0x7fc
.globl __global_pointer$
.hidden __global_pointer$
__global_pointer$:
  .word 0
.section .low,"aw"
low:
  .word 0

#--- no-gp.s
.globl _start
_start:
  lla a0, foo
.section .sdata,"aw"
foo:
  .word 0

#--- lds
SECTIONS {
  .low 0x400 : { *(.low) }
  .text 0x10000 : { *(.text) }
  .sdata 0x200000 : { *(.sdata) }
  .got 0x200900 : { *(.got) }
}
