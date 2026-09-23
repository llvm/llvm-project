## Code-size estimation must emit the AUIPC labels referenced by PCREL_LO12 fixups.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --force-patch -o %t.bolt 2>&1 | FileCheck %s
# RUN: llvm-objdump -d --section=.bolt.org.text %t.bolt | FileCheck %s --check-prefix=PATCH

# CHECK: BOLT-INFO: enabling relocation mode
# CHECK-NOT: could not find corresponding %pcrel_hi
# PATCH: auipc t0,
# PATCH-NEXT: addi t0, t0,
# PATCH-NEXT: jr t0

  .text
  .globl _start
  .type _start,@function
_start:
  li a0, 0
  li a7, 93
  ecall
  .reloc 0, R_RISCV_NONE
  .size _start, .-_start
