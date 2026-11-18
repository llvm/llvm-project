# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=NOREL
# RUN: ld.lld --btf-merge --entry=0 %t.o -o %t
# RUN: llvm-readelf -x .BTF %t | FileCheck %s

## Model GCC's zero offsets with no .BTF relocations. Recover both a global
## and a local offset from .bss, but drop the entry whose symbol is in .data.
# NOREL: There are no relocations in this file.
# CHECK:      0x00000050 01000000 01000000 0f000000 0200000f
# CHECK-NEXT: 0x00000060 0c000000 02000000 04000000 04000000
# CHECK-NEXT: 0x00000070 03000000 08000000 04000000

.bss
  .zero 4
.globl g
.type g,@object
g: .zero 4
.type s,@object
s: .zero 4
.data
.globl other
.type other,@object
other: .long 1

.section .BTF,"",@progbits
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lstr - .Ltypes, .Lstr - .Ltypes, .Lend - .Lstr
.Ltypes:
.long 1, 0x01000000, 4, 32
.long 5, 0x0e000000, 1, 1
.long 7, 0x0e000000, 1, 0
.long 9, 0x0e000000, 1, 1
.long 15, 0x0f000003, 0, 2, 0, 4, 3, 0, 4, 4, 0, 4
.Lstr:
.asciz "", "int", "g", "s", "other", ".bss"
.Lend:
