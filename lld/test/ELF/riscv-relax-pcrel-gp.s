# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax a.s -o rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax a.s -o rv64.o
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv32.o lds -o rv32
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv64.o lds -o rv64
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases rv32 | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases rv64 | FileCheck %s

# CHECK:      <_start>:
# CHECK-NEXT: addi a1, gp, -0x800
# CHECK-NEXT: lw a2, -0x800(gp)
# CHECK-NEXT: sw a2, -0x800(gp)
# CHECK:      <.Lhi1>:
# CHECK-NEXT: addi a1, gp, 0x7ff
# CHECK-NEXT: lw a2, 0x7ff(gp)
# CHECK-NEXT: sw a2, 0x7ff(gp)
# CHECK:      <.Lhi2>:
# CHECK-NEXT: addi a1, gp, -0x7fc
# CHECK-NEXT: lw a2, -0x7fc(gp)
# CHECK-NEXT: sw a2, -0x7fc(gp)
# CHECK:      <.Lhi3>:
# CHECK-NEXT: auipc a0, {{.*}}
# CHECK-NEXT: addi a1, a0, {{.*}}
# CHECK-NEXT: lw a2, {{.*}}(a0)
# CHECK-NEXT: sw a2, {{.*}}(a0)
# CHECK:      <.Lhi4>:
# CHECK-NEXT: addi a1, zero, 0x0
# CHECK-NEXT: lw a2, 0x0(zero)
# CHECK-NEXT: sw a2, 0x0(zero)
# CHECK:      <.Lhi5>:
# CHECK-NEXT: addi a1, zero, 0x7ff
# CHECK-NEXT: lw a2, 0x7ff(zero)
# CHECK-NEXT: sw a2, 0x7ff(zero)
# CHECK:      <.Lhi6>:
# CHECK-NEXT: addi a1, zero, -0x800
# CHECK-NEXT: lw a2, -0x800(zero)
# CHECK-NEXT: sw a2, -0x800(zero)
# CHECK:      <.Lhi7>:
# CHECK-NEXT: auipc a0, {{.*}}
# CHECK-NEXT: addi a1, a0, {{.*}}
# CHECK-NEXT: lw a2, {{.*}}(a0)
# CHECK-NEXT: sw a2, {{.*}}(a0)
# CHECK:      <.Lpcrel_hi0>:
# CHECK-NEXT: addi a3, gp, -0x800
# CHECK:      <.Lpcrel_hi1>:
# CHECK-NEXT: addi a4, gp, 0x7ff

#--- a.s
.macro access target
.Lhi\@:
  auipc a0, %pcrel_hi(\target)
  addi a1, a0, %pcrel_lo(.Lhi\@)
  lw a2, %pcrel_lo(.Lhi\@)(a0)
  sw a2, %pcrel_lo(.Lhi\@)(a0)
.endm

.globl _start
_start:
  access foo
  access bar
  access foo+4
  access outside
  access zero
  access positive
  access negative
  access too_negative
  la a3, foo
  lla a4, bar

.section .sdata,"aw"
foo:
  .space 4095
bar:
  .byte 0
outside:
  .byte 0

#--- lds
SECTIONS {
  .text 0x10000 : { *(.text) }
  .sdata 0x200000 : { *(.sdata) }
}
zero = 0;
positive = 0x7ff;
negative = -0x800;
too_negative = -0x801;
