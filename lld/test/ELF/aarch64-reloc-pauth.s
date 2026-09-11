# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared2.s -o a.o
# RUN: ld.lld -shared a.o -soname=so -o a.so
# RUN: llvm-mc -filetype=obj -triple=aarch64 main.s -o main.o

# RUN: ld.lld -pie main.o a.so -o main
# RUN: llvm-readobj -r main | FileCheck --check-prefix=UNPACKED %s

# UNPACKED:          Section ({{.+}}) .rela.dyn {
# UNPACKED-NEXT:       0x30470 R_AARCH64_AUTH_RELATIVE - 0x1
# UNPACKED-NEXT:       0x30478 R_AARCH64_AUTH_RELATIVE - 0x30472
# UNPACKED-NEXT:       0x30480 R_AARCH64_AUTH_RELATIVE - 0xFFFFFFFFFFFFFFFD
# UNPACKED-NEXT:       0x30488 R_AARCH64_AUTH_RELATIVE - 0x12345678
# UNPACKED-NEXT:       0x30490 R_AARCH64_AUTH_RELATIVE - 0x123456789A
# UNPACKED-NEXT:       0x30498 R_AARCH64_AUTH_RELATIVE - 0xFFFFFFEDCBA98766
# UNPACKED-NEXT:       0x304A0 R_AARCH64_AUTH_RELATIVE - 0x8003046F
# UNPACKED-NEXT:       0x304B9 R_AARCH64_AUTH_RELATIVE - 0x4
# UNPACKED-NEXT:       0x304C2 R_AARCH64_AUTH_RELATIVE - 0x30475
# UNPACKED-NEXT:       0x304A8 R_AARCH64_AUTH_ABS64 zed2 0x1111
# UNPACKED-NEXT:       0x304B0 R_AARCH64_AUTH_ABS64 bar2 0x0
# UNPACKED-NEXT:     }

# RUN: ld.lld main.o a.so -o main.nopie
# RUN: llvm-readobj -r main.nopie | FileCheck --check-prefix=NOPIE %s

# NOPIE:      Section ({{.+}}) .rela.dyn {
# NOPIE:        0x230460 R_AARCH64_AUTH_RELATIVE - 0x200001
# NOPIE-NEXT:   0x230468 R_AARCH64_AUTH_RELATIVE - 0x230462
# NOPIE-NEXT:   0x230470 R_AARCH64_AUTH_RELATIVE - 0x1FFFFD
# NOPIE-NEXT:   0x230478 R_AARCH64_AUTH_RELATIVE - 0x12545678
# NOPIE-NEXT:   0x230480 R_AARCH64_AUTH_RELATIVE - 0x123476789A
# NOPIE-NEXT:   0x230488 R_AARCH64_AUTH_RELATIVE - 0xFFFFFFEDCBC98766
# NOPIE-NEXT:   0x230490 R_AARCH64_AUTH_RELATIVE - 0x8023045F
# NOPIE-NEXT:   0x2304A9 R_AARCH64_AUTH_RELATIVE - 0x200004
# NOPIE-NEXT:   0x2304B2 R_AARCH64_AUTH_RELATIVE - 0x230465
# NOPIE-NEXT:   0x230498 R_AARCH64_AUTH_ABS64 zed2 0x1111
# NOPIE-NEXT:   0x2304A0 R_AARCH64_AUTH_ABS64 bar2 0x0
# NOPIE-NEXT: }

# RUN: ld.lld -pie -z pack-relative-relocs main.o a.so -o main.pie
# RUN: llvm-readelf -S -d -r -x .dynamic -x .test main.pie | FileCheck --check-prefixes=RELR,HEX %s

# RELR:      Section Headers:
# RELR-NEXT: Name Type Address Off Size ES Flg Lk Inf Al
# RELR:      .rela.dyn RELA {{0*}}[[ADDR1:.+]] {{0*}}[[ADDR1]] 000090 18 A 1 0 8
# RELR:      .relr.auth.dyn AARCH64_AUTH_RELR {{0*}}[[ADDR2:.+]] {{0*}}[[ADDR2]] 000018 08 A 0 0 8

# RELR:      Dynamic section at offset {{.+}} contains 16 entries
# RELR:      0x0000000000000007 (RELA)                 0x[[ADDR1]]
# RELR-NEXT: 0x0000000000000008 (RELASZ)               144 (bytes)
# RELR-NEXT: 0x0000000000000009 (RELAENT)              24 (bytes)
# RELR-NEXT: 0x0000000070000012 (AARCH64_AUTH_RELR)    0x[[ADDR2]]
# RELR-NEXT: 0x0000000070000011 (AARCH64_AUTH_RELRSZ)  24 (bytes)
# RELR-NEXT: 0x0000000070000013 (AARCH64_AUTH_RELRENT) 8 (bytes)
# RELR:      0x0000000000000000 (NULL)                 0x0
# RELR-EMPTY:

## Decoded SHT_RELR section is same as UNPACKED,
## but contains only the relative relocations.
## Any relative relocations with odd offset or value wider than 32 bits stay in SHT_RELA.

# RELR:      Relocation section '.rela.dyn' at offset 0x[[ADDR1]] contains 6 entries:
# RELR-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# RELR-NEXT: 0000000000030460  0000000000000411 R_AARCH64_AUTH_RELATIVE           123456789a
# RELR-NEXT: 0000000000030468  0000000000000411 R_AARCH64_AUTH_RELATIVE           ffffffedcba98766
# RELR-NEXT: 0000000000030470  0000000000000411 R_AARCH64_AUTH_RELATIVE           8003043f
# RELR-NEXT: 0000000000030489  0000000000000411 R_AARCH64_AUTH_RELATIVE           4
# RELR-NEXT: 0000000000030478  0000000100000244 R_AARCH64_AUTH_ABS64   0000000000000000 zed2 + 1111
# RELR-NEXT: 0000000000030480  0000000200000244 R_AARCH64_AUTH_ABS64   0000000000000000 bar2 + 0
# RELR-EMPTY:
# RELR-NEXT: Relocation section '.relr.auth.dyn' at offset 0x[[ADDR2]] contains 5 entries:
# RELR-NEXT: Index: Entry Address Symbolic Address
# RELR-NEXT: 0000: 0000000000030440 0000000000030440 $d
# RELR-NEXT: 0001: 000000000000000f 0000000000030448 $d + 0x8
# RELR-NEXT:  0000000000030450 $d + 0x10
# RELR-NEXT:  0000000000030458 $d + 0x18
# RELR-NEXT: 0002: 0000000000030492 0000000000030492 $d + 0x52

# HEX:      Hex dump of section '.dynamic':
# HEX:      0x00020430 00000000 00000000 00000000 00000000
# HEX-EMPTY:

# HEX:      Hex dump of section '.test':
# HEX-NEXT: 0x00030440 01000000 2a000020 42040300 2b000000
##                     ^^^^^^^^ Implicit val = 1 = __ehdr_start + 1
##                              ^^^^ Discr = 42
##                                    ^^ Key (bits 5..6) = DA
##                                       ^^^^^^^^ Implicit val = 0x30442 = 0x30440 + 2 = .test + 2
##                                                ^^^^ Discr = 43
##                                                      ^^ Key (bits 5..6) = IA
# HEX-NEXT: 0x00030450 fdffffff 2c000080 78563412 2d000020
##                     ^^^^^^^^ Implicit val = -3 = __ehdr_start - 3
##                              ^^^^ Discr = 44
##                                    ^^ Key (bits 5..6) = IA
##                                    ^^ Addr diversity (bit 7) = true
##                                       ^^^^^^^^ Implicit val = 0x12345678 = __ehdr_start + 0x12345678
##                                                ^^^^ Discr = 45
##                                                      ^^ Key (bits 5..6) = DA
# HEX-NEXT: 0x00030460 00000000 2e000020 00000000 2f000020
##                     ^^^^^^^^ No implicit val (rela reloc due val wider than 32 bits)
##                              ^^^^ Discr = 46
##                                    ^^ Key (bits 5..6) = DA
##                                       ^^^^^^^^ No implicit val (rela reloc due to val wider than 32 bits)
##                                                ^^^^ Discr = 47
##                                                      ^^ Key (bits 5..6) = DA
# HEX-NEXT: 0x00030470 00000000 30000020 00000000 31000020
##                     ^^^^^^^^ No implicit val (rela reloc due val wider than 32 bits)
##                              ^^^^ Discr = 48
##                                    ^^ Key (bits 5..6) = DA
##                                       ^^^^^^^^ No implicit val (rela reloc due to a preemptible symbol)
##                                                ^^^^ Discr = 49
##                                                      ^^ Key (bits 5..6) = DA
# HEX-NEXT: 0x00030480 00000000 32000000 77000000 00330000
##                     ^^^^^^^^ No implicit val (rela reloc due to a preemptible symbol)
##                              ^^^^ Discr = 50
##                                    ^^ Key (bits 5..6) = IA
##                                         ^^^^^^ ^^ No implicit val (rela reloc due to odd offset)
##                                                  ^^^^ Discr = 51
# HEX-NEXT: 0x00030490 20774504 03003400 0020{{\ }}
##                     ^^ Key (bits 5..6) = DA
##                         ^^^^ ^^^^ Implicit val = 0x30445 = 0x30440 + 5 = .test + 5
##                                  ^^^^ Discr = 52
##                                         ^^ Key (bits 5..6) = DA

#--- main.s

.section .test, "aw"
.p2align 3
.quad (__ehdr_start + 1)@AUTH(da,42)
.quad (.test + 2)@AUTH(ia,43)
.quad (__ehdr_start - 3)@AUTH(ia,44,addr)
.quad (__ehdr_start + 0x12345678)@AUTH(da,45)
## Addend wider than 32 bits, not enough room for storing implicitly, would go to rela
.quad (__ehdr_start + 0x123456789A)@AUTH(da,46)
## Negative addend wider than 32 bits, not enough room for storing implicitly, would go to rela
.quad (__ehdr_start - 0x123456789A)@AUTH(da,47)
## INT32_MAX plus non-zero .test is wider than 32 bits, not enough room for storing implicitly, would go to rela
.quad (.test + 0x7FFFFFFF)@AUTH(da,48)
.quad (zed2 + 0x1111)@AUTH(da,49)
.quad bar2@AUTH(ia,50)
.byte 0x77
.quad (__ehdr_start + 4)@AUTH(da,51)
.byte 0x77
.quad (.test + 5)@AUTH(da,52)

#--- empty-rela.s

## .relr.auth.dyn relocations that do not fit 32 bits are moved to .rela.dyn.
## If this scenario does not happen, .rela.dyn will remain empty,
## but removeUnusedSyntheticSections fails to remove the section.

# RUN: llvm-mc -filetype=obj -triple=aarch64 empty-rela.s -o empty-rela.o
# RUN: ld.lld -pie -z pack-relative-relocs empty-rela.o -o empty-rela
# RUN: llvm-readelf -S -d -r -x.dynamic empty-rela | FileCheck --check-prefixes=EMPTY-RELA %s
# RUN: ld.lld -r -z pack-relative-relocs empty-rela.o -o empty-rela.ro
# RUN: llvm-readelf -S empty-rela.ro | FileCheck --check-prefixes=EMPTY-RELA-RO %s

# EMPTY-RELA:      Section Headers:
# EMPTY-RELA-NEXT: Name Type Address Off Size ES Flg Lk Inf Al
# EMPTY-RELA:      .rela.dyn RELA {{0*}}[[ADDR1:.+]] {{0*}}[[ADDR1]] 000000 18 A 0 0 8
# EMPTY-RELA:      .relr.auth.dyn AARCH64_AUTH_RELR {{0*}}[[ADDR2:.+]] {{0*}}[[ADDR2]] 000008 08 A 0 0 8

# EMPTY-RELA:      Dynamic section at offset {{.+}} contains 12 entries
# EMPTY-RELA-NOT:  (RELR)
# EMPTY-RELA-NOT:  (RELRSZ)
# EMPTY-RELA-NOT:  (RELRENT)
# EMPTY-RELA-NOT:  (RELA)
# EMPTY-RELA-NOT:  (RELASZ)
# EMPTY-RELA-NOT:  (RELAENT)
# EMPTY-RELA:      0x0000000070000012 (AARCH64_AUTH_RELR) 0x[[ADDR2]]
# EMPTY-RELA-NEXT: 0x0000000070000011 (AARCH64_AUTH_RELRSZ) 8 (bytes)
# EMPTY-RELA-NEXT: 0x0000000070000013 (AARCH64_AUTH_RELRENT) 8 (bytes)
# EMPTY-RELA:      0x0000000000000000 (NULL) 0x0
# EMPTY-RELA-EMPTY:

# EMPTY-RELA:      Relocation section '.rela.dyn' at offset {{.+}} contains 0 entries:
# EMPTY-RELA-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name
# EMPTY-RELA-EMPTY:
# EMPTY-RELA-NEXT: Relocation section '.relr.auth.dyn' at offset {{.+}} contains 1 entries:
# EMPTY-RELA-NEXT: Index: Entry Address Symbolic Address
# EMPTY-RELA-NEXT: 0000: 0000000000030310 0000000000030310 $d

## The last 16 bytes are the terminating NULL entry; no stale padding follows.
# EMPTY-RELA:      Hex dump of section '.dynamic':
# EMPTY-RELA:      0x00020300 00000000 00000000 00000000 00000000
# EMPTY-RELA-EMPTY:

# EMPTY-RELA-RO-NOT: .rela.dyn

.section .test, "aw"
.p2align 3
.quad (.test + 0x12345678)@AUTH(da,42)

#--- dynamic-section-growth.s

## .relr.auth.dyn relocations that do not fit 32 bits are moved to .rela.dyn.
## If some relocations are moved to the previously empty .rela.dyn and some stay
## in .relr.auth.dyn, the dynamic section needs to contain tags for both these
## sections and have enough space for storing these tags.

# RUN: llvm-mc -filetype=obj -triple=aarch64 dynamic-section-growth.s -o dynamic-section-growth.o
# RUN: ld.lld -shared -z pack-relative-relocs dynamic-section-growth.o -o dynamic-section-growth.so
# RUN: llvm-readelf -S -d -r -x.dynamic dynamic-section-growth.so | FileCheck --check-prefix=DYN-GROW %s

# DYN-GROW:     .rela.dyn         RELA            0000000000000248 000248 000018 18   A  0   0  8
# DYN-GROW:     .relr.auth.dyn    AARCH64_AUTH_RELR 0000000000000260 000260 000008 08   A  0   0  8
# DYN-GROW:     .dynamic          DYNAMIC         0000000000020268 000268 0000d0 10  WA  4   0  8

# DYN-GROW:      Dynamic section at offset {{.+}} contains 13 entries:
# DYN-GROW:      0x0000000000000007 (RELA)                 0x248
# DYN-GROW-NEXT: 0x0000000000000008 (RELASZ)               24 (bytes)
# DYN-GROW-NEXT: 0x0000000000000009 (RELAENT)              24 (bytes)
# DYN-GROW-NEXT: 0x0000000070000012 (AARCH64_AUTH_RELR)    0x260
# DYN-GROW-NEXT: 0x0000000070000011 (AARCH64_AUTH_RELRSZ)  8 (bytes)
# DYN-GROW-NEXT: 0x0000000070000013 (AARCH64_AUTH_RELRENT) 8 (bytes)
# DYN-GROW:      0x0000000000000000 (NULL)                 0x0
# DYN-GROW-EMPTY:

# DYN-GROW:      Relocation section '.rela.dyn' at offset 0x248 contains 1 entries:
# DYN-GROW-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# DYN-GROW-NEXT: 0000000000030340  0000000000000411 R_AARCH64_AUTH_RELATIVE           100030338
# DYN-GROW-EMPTY:

# DYN-GROW:      Relocation section '.relr.auth.dyn' at offset 0x260 contains 1 entries:
# DYN-GROW-NEXT: Index: Entry            Address           Symbolic Address
# DYN-GROW-NEXT: 0000:  0000000000030338 0000000000030338  foo
# DYN-GROW-EMPTY:

# DYN-GROW:      Hex dump of section '.dynamic':
# DYN-GROW:      0x00020328 00000000 00000000 00000000 00000000
# DYN-GROW-EMPTY:

.data
.balign 8
foo:
## Can stay in .relr.auth.dyn
.quad foo@AUTH(da,42)
## Will be moved to .rela.dyn
.quad foo+0x100000000@AUTH(da,42)

#--- dynamic-section-shrink.s
## If all .relr.auth.dyn entries are moved to a non-empty .rela.dyn, the
## AARCH64_AUTH_RELR* tags are dropped and .dynamic shrinks. Check the section
## header size: -d output stops at the first NULL tag and hides stale padding.
## The emptied .relr.auth.dyn is retained (removeUnusedSyntheticSections runs
## before the move).

# RUN: llvm-mc -filetype=obj -triple=aarch64 dynamic-section-shrink.s -o dynamic-section-shrink.o
# RUN: ld.lld -shared -z pack-relative-relocs dynamic-section-shrink.o -o dynamic-section-shrink.so
# RUN: llvm-readelf -S -d -r dynamic-section-shrink.so | FileCheck --check-prefix=DYN-SHRINK %s

# DYN-SHRINK:     .rela.dyn         RELA            0000000000000248 000248 000030 18   A  1   0  8
# DYN-SHRINK:     .relr.auth.dyn    AARCH64_AUTH_RELR 0000000000000278 000278 000000 08   A  0   0  8
# DYN-SHRINK:     .dynamic          DYNAMIC         0000000000020280 000280 0000a0 10  WA  4   0  8

# DYN-SHRINK:      Dynamic section at offset {{.+}} contains 10 entries:
# DYN-SHRINK:      0x0000000000000007 (RELA)    0x248
# DYN-SHRINK-NEXT: 0x0000000000000008 (RELASZ)  48 (bytes)
# DYN-SHRINK-NEXT: 0x0000000000000009 (RELAENT) 24 (bytes)
# DYN-SHRINK-NOT:  (AARCH64_AUTH_RELR
# DYN-SHRINK:      0x0000000000000000 (NULL)    0x0
# DYN-SHRINK-EMPTY:

# DYN-SHRINK:      Relocation section '.rela.dyn' at offset 0x248 contains 2 entries:
# DYN-SHRINK:      Relocation section '.relr.auth.dyn' at offset 0x278 contains 0 entries:

.data
.balign 8
foo:
## Will be moved to .rela.dyn
.quad foo+0x100000000@AUTH(da,42)

## Alignment 1 forces this into .rela.dyn up-front.
.section .data.rel.ro, "aw"
.quad foo@AUTH(da,42)

#--- rela-iplt-end.s
# RUN: llvm-mc -filetype=obj -triple=aarch64 rela-iplt-end.s -o rela-iplt-end.o
# RUN: ld.lld -z pack-relative-relocs rela-iplt-end.o -o rela-iplt-end
# RUN: llvm-readelf -S -s rela-iplt-end | FileCheck --check-prefix=IPLT-END %s

## Ensure that the end address covers 0x30 bytes for both relocations.
# IPLT-END:      .rela.dyn         RELA            00000000002001c8 0001c8 000030 18   A  0   0  8
# IPLT-END:      00000000002001c8     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_start
# IPLT-END-NEXT: 00000000002001f8     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_end

adrp x0, __rela_iplt_start
adrp x0, __rela_iplt_end

.data
.balign 8
foo:
## Will be moved to .rela.dyn
.quad foo+0x100000000@AUTH(da,42)

## Extra relocation in a section with alignment 1 to force it into .rela.dyn up-front.
.section .data.rel.ro, "aw"
.quad foo@AUTH(da,42)

#--- rela-iplt-start.s
# RUN: llvm-mc -filetype=obj -triple=aarch64 rela-iplt-start.s -o rela-iplt-start.o
# RUN: ld.lld -z pack-relative-relocs rela-iplt-start.o -o rela-iplt-start
# RUN: llvm-readelf -S -s rela-iplt-start | FileCheck --check-prefix=IPLT-START %s

## Ensure that the __rela_iplt* addresses are properly set given that initially .rela.dyn is empty (before moving relocs from .relr.auth.dyn).
# IPLT-START:      .rela.dyn         RELA            0000000000200158 000158 000018 18   A  0   0  8
# IPLT-START:      0000000000200158     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_start
# IPLT-START-NEXT: 0000000000200170     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_end

adrp x0, __rela_iplt_start
adrp x0, __rela_iplt_end

.data
.balign 8
foo:
## Will be moved to .rela.dyn
.quad foo+0x100000000@AUTH(da,42)
