# REQUIRES: mips
# Check R_MIPS_32 relocation calculation.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x10000; .data ALIGN(0x1000) : { *(.data) } \
# RUN:       }" > %t.script

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-be.o
# RUN: ld.lld -shared %t-be.o -script %t.script -o %t-be.so
# RUN: llvm-objdump -t -s %t-be.so | FileCheck --check-prefixes=SYM,BE %s
# RUN: llvm-readelf -r -s --dynamic-table -A %t-be.so \
# RUN:   | FileCheck -check-prefix=REL %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s -o %t-el.o
# RUN: ld.lld -shared %t-el.o -script %t.script -o %t-el.so
# RUN: llvm-objdump -t -s %t-el.so | FileCheck --check-prefixes=SYM,EL %s
# RUN: llvm-readelf --dynamic-table -r -s -A %t-el.so \
# RUN:   | FileCheck -check-prefix=REL %s

  .data
  .globl v2
v1:
  .word v2+4 # R_MIPS_32 target v2 addend 4
v2:
  .word v1   # R_MIPS_32 target v1 addend 0

# SYM: SYMBOL TABLE:
# SYM: 00011000 l .data  00000000 v1

# BE: Contents of section .data:
# BE-NEXT: {{.*}} 00000004 00011000
#                 ^-- v2+4 ^-- v1

# EL: Contents of section .data:
# EL-NEXT: {{.*}} 04000000 00100100
#                 ^-- v2+4 ^-- v1

# REL: Dynamic section
# REL:     (RELSZ)    16
# REL:     (RELENT)    8
# REL-NOT: (RELCOUNT)

# REL: Relocation section
# REL:      {{.*}} R_MIPS_REL32 [[V2:[0-9a-f]+]]
# REL-NEXT: {{.*}} R_MIPS_REL32 {{$}}

# REL: Symbol table
# REL: {{.*}}: [[V2]] {{.*}} v2

# REL: Global entries
# REL: {{.*}} -32744(gp) [[V2]] {{.*}} v2
