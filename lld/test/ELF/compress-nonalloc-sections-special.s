# REQUIRES: x86, zlib

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -pie %t.o --compress-nonalloc-sections .strtab=zlib --compress-nonalloc-sections .symtab=zlib -o %t
# RUN: llvm-readelf -Ss -x .strtab %t 2>&1 | FileCheck %s

# CHECK:      nonalloc0  PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00     0 0  1
# CHECK:      .symtab    SYMTAB   0000000000000000 [[#%x,]] [[#%x,]] 18  C 12 3  1
# CHECK-NEXT: .shstrtab  STRTAB   0000000000000000 [[#%x,]] [[#%x,]] 00     0 0  1
# CHECK-NEXT: .strtab    STRTAB   0000000000000000 [[#%x,]] [[#%x,]] 00  C  0 0  1

## TODO Add compressed SHT_STRTAB/SHT_SYMTAB support to llvm-readelf
# CHECK:      warning: {{.*}}: unable to get the string table for the SHT_SYMTAB section: SHT_STRTAB string table section

# CHECK:      Hex dump of section '.strtab':
# CHECK-NEXT: 01000000 00000000 1a000000 00000000
# CHECK-NEXT: 01000000 00000000 {{.*}}

.globl _start, g0, g1
_start:
l0:
g0:
g1:

.section nonalloc0,""
.quad .text+1
.quad .text+2
