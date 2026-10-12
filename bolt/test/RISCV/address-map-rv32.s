## Check that BOLT parses its internal address map correctly for a 32-bit
## target. Address-map entries contain 64-bit values even when the target code
## pointer size is 32 bits.

# REQUIRES: asserts

# RUN: llvm-mc -triple=riscv32 -mattr=+c -filetype=obj %s -o %t.o
# RUN: ld.lld -q %t.o -o %t
# RUN: llvm-bolt --enable-bat %t -o %t.bolt

  .text
  .globl _start
  .type _start, @function
  .p2align 1
_start:
  nop
1:
  j 1b
  .size _start, .-_start
