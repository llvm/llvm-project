# Check R_MIPS_32 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-be.o
# RUN: ld.lld2 %t-be.o -o %t-be.exe
# RUN: llvm-objdump -t %t-be.exe | FileCheck %s
# RUN: llvm-objdump -s %t-be.exe | FileCheck -check-prefix=BE %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s -o %t-el.o
# RUN: ld.lld2 %t-el.o -o %t-el.exe
# RUN: llvm-objdump -t %t-el.exe | FileCheck %s
# RUN: llvm-objdump -s %t-el.exe | FileCheck -check-prefix=EL %s

# REQUIRES: mips

  .globl  __start
__start:
  nop

  .data
  .type  v1,@object
  .size  v1,4
v1:
  .word 0

  .globl v2
  .type  v2,@object
  .size  v2,8
v2:
  .word v2+4 # R_MIPS_32 target v2 addend 4
  .word v1   # R_MIPS_32 target v1 addend 0

# CHECK: SYMBOL TABLE:
# CHECK: 00030000 l       .data           00000004 v1
# CHECK: 00030004 g       .data           00000008 v2

# BE: Contents of section .data:
# BE-NEXT: 30000 00000000 00030008 00030000
#                         ^-- v2+4 ^-- v1

# EL: Contents of section .data:
# EL-NEXT: 30000 00000000 08000300 00000300
#                         ^-- v2+4 ^-- v1
