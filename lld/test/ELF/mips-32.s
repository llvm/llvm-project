# REQUIRES: mips
# Check R_MIPS_32 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-be.o
# RUN: ld.lld -shared %t-be.o -o %t-be.so
# RUN: llvm-objdump -t -s %t-be.so | FileCheck -check-prefixes=SYM,BE %s
# RUN: llvm-readelf -r -s --dynamic-table --mips-plt-got %t-be.so \
# RUN:   | FileCheck -check-prefix=REL %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s -o %t-el.o
# RUN: ld.lld -shared %t-el.o -o %t-el.so
# RUN: llvm-objdump -t -s %t-el.so | FileCheck -check-prefixes=SYM,EL %s
# RUN: llvm-readelf -r -s --dynamic-table --mips-plt-got %t-el.so \
# RUN:   | FileCheck -check-prefix=REL %s

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

# BE: Contents of section .data:
# BE-NEXT: {{.*}} 00000000 00000004 00020000
#                          ^-- v2+4 ^-- v1

# EL: Contents of section .data:
# EL-NEXT: {{.*}} 00000000 04000000 00000200
#                          ^-- v2+4 ^-- v1

# SYM: SYMBOL TABLE:
# SYM: 00020000 l     O .data           00000004 v1

# Relocation section
# REL:      {{.*}} R_MIPS_REL32
# REL-NEXT: {{.*}} R_MIPS_REL32 [[V2:[0-9a-f]+]]

# REL: Symbol table
# REL: {{.*}} [[V2]] {{.*}} v2

# Dynamic section
# REL:     (RELSZ)    16
# REL:     (RELENT)    8
# REL-NOT: (RELCOUNT)

# REL: Global entries
# REL: {{.*}} -32744(gp) [[V2]] {{.*}} v2
