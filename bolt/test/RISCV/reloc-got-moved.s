## Check that the RV64 R_RISCV_GOT_HI20/%pcrel_lo pair is rebuilt when the
## matching low instruction is not immediately after AUIPC.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
# RUN: ld.lld -q -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt -reorder-functions=cdsort --check-encoding
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# CHECK: Disassembly of section .text:
# CHECK: <_start>:
# CHECK-NEXT: auipc a0, 0xffc12
# CHECK-NEXT: li a1, 0x7
# CHECK-NEXT: li a2, 0x9
# CHECK-NEXT: ld a0, 0x1e0(a0)
# CHECK-NEXT: ret

  .data
  .p2align 12
  .globl d
d:
  .dword 0

  .text
  .globl _start
  .type _start, @function
_start:
  nop
1:
  auipc a0, %got_pcrel_hi(d)
  addi a1, zero, 7
  addi a2, zero, 9
  ld a0, %pcrel_lo(1b)(a0)
  ret
  .reloc 0, R_RISCV_NONE
  .size _start, .-_start
