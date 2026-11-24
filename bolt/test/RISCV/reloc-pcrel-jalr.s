## Check that disassembly keeps the PC-relative operand of an indirect JALR.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --print-disasm --print-only=_start -o %t.bolt \
# RUN:   | FileCheck %s

# CHECK-LABEL: Binary Function "_start" after disassembly {
# CHECK:       auipc t0, %pcrel_hi(_start) # Label: [[HI:\.Ltmp[0-9]+]]
# CHECK-NEXT:  jalr t0, %pcrel_lo([[HI]])(t0)

  .text
  .globl _start
  .type _start,@function
_start:
  auipc t0, %pcrel_hi(_start)
  jalr t0, %pcrel_lo(_start)(t0)
  .size _start, .-_start
