# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o %t.32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.64.o
# RUN: ld.lld %t.32.o --defsym low=0xfffff800 --defsym high=0xffffffff -o %t.32
# RUN: ld.lld %t.64.o --defsym low=0xfffffffffffff800 --defsym high=0xffffffffffffffff -o %t.64
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.32 | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.64 | FileCheck %s

# CHECK:      <_start>:
# CHECK-NEXT: addi a0, zero, -0x800
# CHECK-NEXT: lw a1, -0x800(zero)
# CHECK-NEXT: sw a1, -0x800(zero)
# CHECK-NEXT: addi a0, zero, -0x1
# CHECK-NEXT: lw a1, -0x1(zero)
# CHECK-NEXT: sw a1, -0x1(zero)

.globl _start
_start:
  lui a0, %hi(low)
  addi a0, a0, %lo(low)
  lw a1, %lo(low)(a0)
  sw a1, %lo(low)(a0)
  lui a0, %hi(high)
  addi a0, a0, %lo(high)
  lw a1, %lo(high)(a0)
  sw a1, %lo(high)(a0)
