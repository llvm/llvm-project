## Check that RV32 R_RISCV_PCREL_LO12 relocations are re-encoded relative to
## the moved AUIPC instruction instead of retaining the input addend.

# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj -o %t.o %s
# RUN: ld.lld -q -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt -reorder-functions=cdsort --check-encoding
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# CHECK: Disassembly of section .text:
# CHECK: <_start>:
# CHECK-NEXT: auipc a0, 0xffc13
# CHECK-NEXT: lw a0, 0x0(a0)
# CHECK-NEXT: ret

  .data
  .p2align 12
  .globl d
d:
  .word 0

  .text
  .globl _start
  .type _start, @function
_start:
  nop
1:
  auipc a0, %pcrel_hi(d)
  lw a0, %pcrel_lo(1b)(a0)
  ret
  .reloc 0, R_RISCV_NONE
  .size _start, .-_start
