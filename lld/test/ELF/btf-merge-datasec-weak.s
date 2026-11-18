# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym=WEAK=1 %s -o %t.weak.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.strong.o
# RUN: ld.lld --btf-merge --entry=0 %t.weak.o %t.strong.o -o %t
# RUN: llvm-readelf -x .BTF %t | FileCheck %s

## Both .data input sections stay live, but only the strong VAR (type 3)
## should have a DATASEC entry. The weak VAR (type 2) is kept as a type.
# CHECK:      0x00000030 01000000 01000000 05000000 0000000e
# CHECK-NEXT: 0x00000040 01000000 01000000 07000000 0100000f
# CHECK-NEXT: 0x00000050 08000000 03000000 04000000 04000000

.ifdef WEAK
.weak g
.else
.globl g
.endif
.data
.type g,@object
g: .long 1

.section .BTF,"",@progbits
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lstr - .Ltypes, .Lstr - .Ltypes, .Lend - .Lstr
.Ltypes:
.long 1, 0x01000000, 4, 32
.long 5, 0x0e000000, 1, 1
.long 7, 0x0f000001, 0, 2, g, 4
.Lstr:
.asciz "", "int", "g", ".data"
.Lend:
