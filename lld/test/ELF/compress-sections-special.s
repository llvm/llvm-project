# REQUIRES: x86, zlib

# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o
# RUN: ld.lld -pie a.o --compress-sections .strtab=zlib --compress-sections .symtab=zlib -o out
# RUN: llvm-readelf -Ss -x .strtab out 2>&1 | FileCheck %s

# CHECK:      nonalloc0  PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00     0 0  1
# CHECK:      .symtab    SYMTAB   0000000000000000 [[#%x,]] [[#%x,]] 18  C 12 3  1
# CHECK-NEXT: .shstrtab  STRTAB   0000000000000000 [[#%x,]] [[#%x,]] 00     0 0  1
# CHECK-NEXT: .strtab    STRTAB   0000000000000000 [[#%x,]] [[#%x,]] 00  C  0 0  1

## TODO Add compressed SHT_STRTAB/SHT_SYMTAB support to llvm-readelf
# CHECK:      warning: {{.*}}: unable to get the string table for the SHT_SYMTAB section: SHT_STRTAB string table section

# CHECK:      Hex dump of section '.strtab':
# CHECK-NEXT: 01000000 00000000 1a000000 00000000
# CHECK-NEXT: 01000000 00000000 {{.*}}

# RUN: not ld.lld -shared a.o --compress-sections .dynstr=zlib 2>&1 | FileCheck %s --check-prefix=ERR-ALLOC
# ERR-ALLOC: error: --compress-sections: section '.dynstr' with the SHF_ALLOC flag cannot be compressed

.globl _start, g0, g1
_start:
l0:
g0:
g1:

.section nonalloc0,""
.quad .text+1
.quad .text+2
