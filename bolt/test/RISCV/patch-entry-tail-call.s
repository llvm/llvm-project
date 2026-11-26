## A long entry trampoline fits in an eight-byte function and preserves t0.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -Ttext=0x10fc0 %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --force-patch --custom-allocation-vma=0x400000 -o %t.bolt
# RUN: llvm-objdump -d --section=.bolt.org.text -M no-aliases %t.bolt | FileCheck %s

# CHECK-LABEL: <_start.org.0>:
# CHECK-NEXT: auipc t1,
# CHECK-NEXT: jalr zero, {{.*}}(t1) <_start>

  .text
  .globl _start
  .type _start,@function
_start:
  li a7, 93
  ecall
  .reloc 0, R_RISCV_NONE
  .size _start, .-_start
