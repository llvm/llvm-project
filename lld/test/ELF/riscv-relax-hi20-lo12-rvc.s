# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax,+c a.s -o rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax,+c a.s -o rv64.o

# RUN: ld.lld rv32.o lds -o rv32
# RUN: ld.lld rv64.o lds -o rv64
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv32 | FileCheck %s
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv64 | FileCheck %s

# CHECK: 0000002c l       .text {{0*}}0 a

# CHECK:      c.lui   a0, 0x1
# CHECK-NEXT: addi    a0, a0, 0x0
# CHECK-NEXT: lw      a0, 0x0(a0)
# CHECK-NEXT: sw      a0, 0x0(a0)
# CHECK-NEXT: c.lui   a0, 0x1f
# CHECK-NEXT: addi    a0, a0, 0x7ff
# CHECK-NEXT: lb      a0, 0x7ff(a0)
# CHECK-NEXT: sb      a0, 0x7ff(a0)
# CHECK-NEXT: lui     a0, 0x20
# CHECK-NEXT: addi    a0, a0, -0x800
# CHECK-NEXT: lw      a0, -0x800(a0)
# CHECK-NEXT: sw      a0, -0x800(a0)
# CHECK-EMPTY:
# CHECK-NEXT: <a>:
# CHECK-NEXT: c.addi a0, 0x1

#--- a.s
.global _start
_start:
  lui a0, %hi(rvc_lui_low)
  addi a0, a0, %lo(rvc_lui_low)
  lw a0, %lo(rvc_lui_low)(a0)
  sw a0, %lo(rvc_lui_low)(a0)
  lui a0, %hi(rvc_lui_high)
  addi a0, a0, %lo(rvc_lui_high)
  lb a0, %lo(rvc_lui_high)(a0)
  sb a0, %lo(rvc_lui_high)(a0)
  lui a0, %hi(norelax)
  addi a0, a0, %lo(norelax)
  lw a0, %lo(norelax)(a0)
  sw a0, %lo(norelax)(a0)
a:
  addi a0, a0, 1

.section .sdata,"aw"
rvc_lui_low:
  .space 124927
rvc_lui_high:
  .byte 0
norelax:
  .word 0

#--- lds
SECTIONS {
  .text : {*(.text) }
  .sdata 0x1000 : { }
}
