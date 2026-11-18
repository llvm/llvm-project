# REQUIRES: x86
## RELA (x86-64) and REL (i386) must produce the same section-relative offsets.
# RUN: split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld --btf-merge --gc-sections --emit-relocs a.o b.o -o out
# RUN: llvm-readelf -S out | FileCheck %s --check-prefix=SECTIONS
# RUN: llvm-readelf -x .BTF out | FileCheck %s

## A relocatable link leaves both BTF blobs and their relocations in one section.
## Leave the COMDAT unreferenced so GC discards it in both links: -r can retarget
## the discarded definition's relocation to the prevailing symbol.
# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym=NO_GROUP_REF=1 a.s -o a-nogroup.o
# RUN: ld.lld --btf-merge --gc-sections a-nogroup.o b.o -o direct
# RUN: ld.lld -r --btf-merge a-nogroup.o b.o -o partial.o
# RUN: ld.lld --btf-merge --gc-sections partial.o -o from-partial
# RUN: llvm-objcopy --dump-section=.BTF=direct.btf direct direct-copy
# RUN: llvm-objcopy --dump-section=.BTF=partial.btf from-partial partial-copy
# RUN: cmp direct.btf partial.btf
# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym=BAD_PADDING=1 a.s -o bad.o
# RUN: ld.lld -r bad.o b.o -o bad-partial.o
# RUN: not ld.lld --btf-merge bad-partial.o -o bad 2>&1 | FileCheck %s --check-prefix=BAD-PADDING

# RUN: ld.lld --btf-merge -T discard.lds a.o b.o -o discard
# RUN: llvm-readelf -x .BTF discard | FileCheck %s
# RUN: llvm-readelf -p .BTF discard | FileCheck %s --check-prefix=RENAMED
# RUN: llvm-mc -filetype=obj -triple=i386 a.s -o a32.o
# RUN: llvm-mc -filetype=obj -triple=i386 b.s -o b32.o
# RUN: llvm-readelf -r a32.o | FileCheck %s --check-prefix=REL
# RUN: ld.lld --btf-merge --gc-sections a32.o b32.o -o out32
# RUN: llvm-readelf -x .BTF out32 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=x86_64 overlap.s -o overlap.o
# RUN: ld.lld --btf-merge overlap.o -o overlap
# RUN: llvm-readelf -x .BTF overlap | FileCheck %s --check-prefix=OVERLAP
# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym=UNALIGNED=1 b.s -o unaligned.o
# RUN: ld.lld -r a-nogroup.o unaligned.o -o mixed.o
# RUN: ld.lld --btf-merge mixed.o -o mixed
# RUN: llvm-mc -filetype=obj -triple=x86_64 empty.s -o empty.o
# RUN: ld.lld --btf-merge -T empty.lds empty.o -o empty
# RUN: llvm-readelf -x .BTF empty | FileCheck %s --check-prefix=EMPTY

## The empty DATASEC header and its entry are both removed.
# EMPTY: 0x00000000 9feb0100 18000000 00000000 38000000
# EMPTY-NEXT: 0x00000010 38000000

## Keep the first of two aliases and the adjacent variable, but omit the empty.
# OVERLAP:      0x00000000 9feb0100 18000000 00000000 74000000
# OVERLAP-NEXT: 0x00000010 74000000 3a000000
# OVERLAP:      0x00000060 01000000 00000000 1c000000 0200000f
# OVERLAP-NEXT: 0x00000070 08000000 02000000 00000000 04000000
# OVERLAP-NEXT: 0x00000080 05000000 04000000 04000000

# SECTIONS: .BTF PROGBITS
# SECTIONS-NOT: .rela.BTF
# BAD-PADDING: error: bad-partial.o:(.BTF): cannot validate .BTF relocations: invalid .BTF magic
# RENAMED:      .xdat
# RENAMED-NEXT: .bss
# REL: Relocation section '.rel.BTF'
# REL: R_386_32 {{.*}} .bss

## Types 1-7: one INT, then six distinct VARs (including the two discarded
## variables). Type 8: .data, size 20, with VARs at 0, 4, and 16. Type 9:
## .bss, size 12, with its static at 8 (a section symbol plus addend).
# CHECK:      Hex dump of section '.BTF':
# CHECK-NEXT: 0x00000000 9feb0100 18000000 00000000 b8000000
# CHECK-NEXT: 0x00000010 b8000000 2d000000 01000000 00000001
# CHECK-NEXT: 0x00000020 04000000 20000000 05000000 0000000e
# CHECK-NEXT: 0x00000030 01000000 00000000 0b000000 0000000e
# CHECK-NEXT: 0x00000040 01000000 01000000 11000000 0000000e
# CHECK-NEXT: 0x00000050 01000000 00000000 17000000 0000000e
# CHECK-NEXT: 0x00000060 01000000 00000000 05000000 0000000e
# CHECK-NEXT: 0x00000070 01000000 00000000 1c000000 0000000e
# CHECK-NEXT: 0x00000080 01000000 01000000 22000000 0300000f
# CHECK-NEXT: 0x00000090 14000000 03000000 00000000 04000000
# CHECK-NEXT: 0x000000a0 02000000 04000000 04000000 06000000
# CHECK-NEXT: 0x000000b0 10000000 04000000 28000000 0100000f
# CHECK-NEXT: 0x000000c0 0c000000 04000000 08000000 04000000

#--- a.s
.globl _start, first, group
.text
_start:
  .long first, bssvar, second
.ifndef NO_GROUP_REF
  .long group
.endif

.data
first: .long 1
local: .long 2
.bss
  .zero 8
bssvar: .zero 4
.section .unused,"aw"
dead: .long 3
.section .groupdata,"awG",@progbits,group,comdat
group: .long 4

.section .BTF,"",@progbits
.p2align 2
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lstr - .Ltypes, .Lstr - .Ltypes, .Lend - .Lstr
.Ltypes:
.long 1, 0x01000000, 4, 32
.long 5, 0x0e000000, 1, 0
.long 11, 0x0e000000, 1, 1
.long 17, 0x0e000000, 1, 0
.long 23, 0x0e000000, 1, 0
## List the .data entries in reverse order.
.long 28, 0x0f000002, 0, 2, local-first, 4, 3, first, 4
.long 34, 0x0f000001, 0, 4, bssvar, 4
.long 39, 0x0f000001, 0, 5, dead, 4
.Lstr:
.asciz "", "int", "local", "first", "bss01", "dead", ".data", ".bss", ".unused"
.Lend:
.ifdef BAD_PADDING
.byte 1
.endif

#--- empty.lds
SECTIONS { .bss.empty : { *(.bss.empty) } }

#--- empty.s
.globl _start, empty
.text
_start: ret
.section .bss.empty,"aw",@nobits
empty:
.section .BTF,"",@progbits
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lstr-.Ltypes, .Lstr-.Ltypes, .Lend-.Lstr
.Ltypes:
.long 1, 0x01000000, 4, 32
.long 0, 0x03000000, 0, 1, 1, 0
.long 5, 0x0e000000, 2, 1
.long 11, 0x0f000001, 0, 3, empty, 0
.Lstr:
.asciz "", "int", "empty", ".bss.empty"
.Lend:

#--- b.s
.globl second, group
.data
.p2align 4
second: .long 5
.section .groupdata,"awG",@progbits,group,comdat
group: .long 6

.section .BTF,"",@progbits
.ifndef UNALIGNED
.p2align 2
.endif
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lstr - .Ltypes, .Lstr - .Ltypes, .Lend - .Lstr
.Ltypes:
.long 1, 0x01000000, 4, 32
.long 5, 0x0e000000, 1, 0
.long 11, 0x0e000000, 1, 1
.long 17, 0x0f000001, 0, 2, second, 4
.long 23, 0x0f000001, 0, 3, group, 4
.Lstr:
.asciz "", "int", "local", "group", ".data", ".groupdata"
.Lend:

#--- discard.lds
SECTIONS {
  .xdat : { *(.data) }
  /DISCARD/ : { *(.unused) }
}

#--- overlap.s
.globl _start
.text
_start: ret
.data
first:
alias: .long 1
empty:
next: .long 2
.section .BTF,"",@progbits
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lstr - .Ltypes, .Lstr - .Ltypes, .Lend - .Lstr
.Ltypes:
.long 1, 0x01000000, 4, 32
.long 5, 0x0e000000, 1, 0
.long 11, 0x0e000000, 1, 0
.long 17, 0x0e000000, 1, 0
.long 23, 0x0e000000, 1, 0
.long 28, 0x0f000004, 0, 2, first, 4, 3, alias, 4, 4, empty, 0, 5, next, 4
.Lstr:
.asciz "", "int", "first", "alias", "empty", "next", ".data"
.Lend:
