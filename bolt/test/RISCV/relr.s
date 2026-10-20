# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld -pie --emit-relocs --pack-dyn-relocs=relr %t.o -o %t.exe
# RUN: llvm-readelf -d %t.exe | FileCheck %s --check-prefix=INPUT
# RUN: llvm-bolt %t.exe -o %t.bolt --reorder-functions=cdsort
# RUN: llvm-readelf -r %t.bolt | FileCheck %s --check-prefix=RELOC
# RUN: llvm-nm %t.bolt | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-readelf -x .data %t.bolt | FileCheck %s --check-prefix=DATA

## Reduced from the RELR-packed clang input: reading relative relocations
## works on RV64, and references to a moved function must be updated.
# INPUT: (RELR)
# RELOC: Relocation section '.relr.dyn'
# RELOC: pointers{{$}}
# RELOC-NEXT: pointers + 0x8
# RELOC-NEXT: pointers + 0x10
# SYMBOL: 0000000000400000 T _start
# DATA: 00004000 00000000 00004000 00000000
# DATA-NEXT: 00004000 00000000

  .text
  .globl _start
  .type _start,@function
_start:
  nop
  ret
  .reloc 0, R_RISCV_NONE
  .size _start, .-_start

  .data
  .p2align 3
  .globl pointers
pointers:
  .quad _start
  .quad _start
  .quad _start
