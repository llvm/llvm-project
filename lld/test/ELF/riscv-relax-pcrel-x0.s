# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o %t.32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.64.o
# RUN: ld.lld -Ttext=0x10000 %t.32.o --defsym low=0xfffff800 --defsym high=0xffffffff --defsym outside=0x800 -o %t.32
# RUN: ld.lld -Ttext=0x10000 %t.64.o --defsym low=0xfffffffffffff800 --defsym high=0xffffffffffffffff --defsym outside=0x800 -o %t.64
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.32 | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.64 | FileCheck %s

# CHECK:      <_start>:
# CHECK-NEXT: addi a1, zero, -0x800
# CHECK-NEXT: lw a2, -0x800(zero)
# CHECK-NEXT: sw a2, -0x800(zero)
# CHECK:      <.Lhi1>:
# CHECK-NEXT: addi a1, zero, -0x1
# CHECK-NEXT: lw a2, -0x1(zero)
# CHECK-NEXT: sw a2, -0x1(zero)
# CHECK:      <.Lhi2>:
# CHECK-NEXT: auipc a0, 0xffff0
# CHECK-NEXT: addi a1, a0, 0x7e8
# CHECK-NEXT: lw a2, 0x7e8(a0)
# CHECK-NEXT: sw a2, 0x7e8(a0)

.macro access target
.Lhi\@:
  auipc a0, %pcrel_hi(\target)
  addi a1, a0, %pcrel_lo(.Lhi\@)
  lw a2, %pcrel_lo(.Lhi\@)(a0)
  sw a2, %pcrel_lo(.Lhi\@)(a0)
.endm
.globl _start
_start:
  access low
  access high
  access outside
