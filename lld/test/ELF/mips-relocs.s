# Check R_MIPS_32 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-be.o
# RUN: ld.lld -shared %t-be.o -o %t-be.so
# RUN: llvm-objdump -t %t-be.so | FileCheck %s
# RUN: llvm-objdump -s %t-be.so | FileCheck -check-prefix=BE %s
# RUN: llvm-readobj -r -dynamic-table %t-be.so | FileCheck -check-prefix=REL %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s -o %t-el.o
# RUN: ld.lld -shared %t-el.o -o %t-el.so
# RUN: llvm-objdump -t %t-el.so | FileCheck %s
# RUN: llvm-objdump -s %t-el.so | FileCheck -check-prefix=EL %s
# RUN: llvm-readobj -r -dynamic-table %t-el.so | FileCheck -check-prefix=REL %s

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

# REL:      Relocations [
# REL-NEXT:   Section (7) .rel.dyn {
# REL-NEXT:     0x30004 R_MIPS_REL32 v2 0x0
# REL-NEXT:     0x30008 R_MIPS_REL32 - 0x0
# REL-NEXT:   }
# REL-NEXT: ]

# REL: DynamicSection [
# REL:   Tag        Type                 Name/Value
# REL:   0x00000012 RELSZ                16 (bytes)
# REL:   0x00000013 RELENT               8 (bytes)
