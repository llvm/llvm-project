# REQUIRES: x86, zlib

# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o
# RUN: ld.lld -pie a.o --compress-sections .strtab=zlib --compress-sections .symtab=zlib -o out
# RUN: llvm-readelf -Ss -x .strtab out 2>&1 | FileCheck %s

# CHECK:      nonalloc0  PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00     0 0  1
# CHECK:      .symtab    SYMTAB   0000000000000000 [[#%x,]] [[#%x,]] 18  C 12 3  1
# CHECK-NEXT: .shstrtab  STRTAB   0000000000000000 [[#%x,]] [[#%x,]] 00     0 0  1
# CHECK-NEXT: .strtab    STRTAB   0000000000000000 [[#%x,]] [[#%x,]] 00  C  0 0  1

# CHECK:      Symbol table '.symtab' contains 8 entries:
# CHECK:      _start

# CHECK:      Hex dump of section '.strtab':
# CHECK-NEXT: 01000000 00000000 5c000000 00000000
# CHECK-NEXT: 01000000 00000000 {{.*}}

# RUN: llvm-nm out | FileCheck %s --check-prefix=NM
# NM:      {{0*}}[[#%x,]] T _start
# NM-NEXT: {{0*}}[[#%x,]] T g0
# NM-NEXT: {{0*}}[[#%x,]] T g1
# NM-NEXT:                U ggggggggggggggggggggggggggggggg0
# NM-NEXT:                U ggggggggggggggggggggggggggggggg1
# NM-NEXT: {{0*}}[[#%x,]] t l0

# RUN: llvm-objcopy out out2
# RUN: llvm-readelf -Ss out2 | FileCheck %s --check-prefix=OBJCOPY
# OBJCOPY:      .symtab    SYMTAB   0000000000000000 [[#%x,]] [[#%x,]] 18    12 3  8
# OBJCOPY-NEXT: .shstrtab  STRTAB   0000000000000000 [[#%x,]] [[#%x,]] 00     0 0  1
# OBJCOPY-NEXT: .strtab    STRTAB   0000000000000000 [[#%x,]] [[#%x,]] 00     0 0  1
# OBJCOPY:      Symbol table '.symtab' contains 8 entries:
# OBJCOPY:      _start

# RUN: not ld.lld -shared a.o --compress-sections .dynstr=zlib 2>&1 | FileCheck %s --check-prefix=ERR-ALLOC
# ERR-ALLOC: error: --compress-sections: section '.dynstr' with the SHF_ALLOC flag cannot be compressed

.globl _start, g0, g1
_start:
l0:
g0:
g1:
.globl ggggggggggggggggggggggggggggggg0
.globl ggggggggggggggggggggggggggggggg1

.section nonalloc0,""
.quad .text+1
.quad .text+2
