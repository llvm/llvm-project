# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared2.s -o %t.so.o
# RUN: ld.lld -shared %t.so.o -soname=so -o %t.so
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o

# RUN: ld.lld -pie -z nopack-relative-relocs %t.o %t.so -o %t2
# RUN: llvm-readobj -r %t2 | FileCheck --check-prefix=UNPACKED %s

# UNPACKED:          Section ({{.+}}) .rela.dyn {
# UNPACKED-NEXT:       0x30440 R_AARCH64_AUTH_RELATIVE - 0x1
# UNPACKED-NEXT:       0x30448 R_AARCH64_AUTH_RELATIVE - 0x2
# UNPACKED-NEXT:       0x30450 R_AARCH64_AUTH_RELATIVE - 0x3
# UNPACKED-NEXT:       0x30458 R_AARCH64_AUTH_RELATIVE - 0x12345678
# UNPACKED-NEXT:       0x30460 R_AARCH64_AUTH_RELATIVE - 0x123456789A
# UNPACKED-NEXT:       0x30479 R_AARCH64_AUTH_RELATIVE - 0x4
# UNPACKED-NEXT:       0x30482 R_AARCH64_AUTH_RELATIVE - 0x5
# UNPACKED-NEXT:       0x30468 R_AARCH64_AUTH_ABS64 zed2 0x0
# UNPACKED-NEXT:       0x30470 R_AARCH64_AUTH_ABS64 bar2 0x0
# UNPACKED-NEXT:     }

# RUN: ld.lld -pie -z pack-relative-relocs %t.o %t.so -o %t2
# RUN: llvm-readelf -S --dynamic-table %t2 | FileCheck --check-prefix=RELR-HEADERS %s

# RELR-HEADERS:      Section Headers:
# RELR-HEADERS-NEXT: Name Type Address Off Size ES Flg Lk Inf Al
# RELR-HEADERS:      .relr.auth.dyn AARCH64_AUTH_RELR {{0*}}[[ADDR:.*]] {{0*}}[[ADDR]] 000018 08 A 0 0 8

# RELR-HEADERS:      Dynamic section at offset 0x310 contains 16 entries
# RELR-HEADERS:      0x0000000070000012 (AARCH64_AUTH_RELR) 0x[[ADDR]]
# RELR-HEADERS-NEXT: 0x0000000070000011 (AARCH64_AUTH_RELRSZ) 24 (bytes)
# RELR-HEADERS-NEXT: 0x0000000070000013 (AARCH64_AUTH_RELRENT) 8 (bytes)

# RUN: llvm-readobj -r --raw-relr %t2 | FileCheck --check-prefix=RAW-RELR %s

## SHT_RELR section contains address/bitmap entries
## encoding the offsets for relative relocation.

# RAW-RELR:           Section ({{.+}}) .relr.auth.dyn {
# RAW-RELR-NEXT:      0x30410
# RAW-RELR-NEXT:      0xF
# RAW-RELR-NEXT:      0x30452
# RAW-RELR-NEXT:      }

# RUN: llvm-readobj -r %t2 | FileCheck --check-prefix=RELR %s

## Decoded SHT_RELR section is same as UNPACKED,
## but contains only the relative relocations.
## Any relative relocations with odd offset stay in SHT_RELA.

# RELR:      Section ({{.+}}) .rela.dyn {
# RELR-NEXT:   0x30430 R_AARCH64_AUTH_RELATIVE - 0x123456789A
# RELR-NEXT:   0x30449 R_AARCH64_AUTH_RELATIVE - 0x4
# RELR-NEXT:   0x30438 R_AARCH64_AUTH_ABS64 zed2 0x0
# RELR-NEXT:   0x30440 R_AARCH64_AUTH_ABS64 bar2 0x0
# RELR-NEXT: }
# RELR-NEXT: Section ({{.+}}) .relr.auth.dyn {
# RELR-NEXT:   0x30410 R_AARCH64_RELATIVE -
# RELR-NEXT:   0x30418 R_AARCH64_RELATIVE -
# RELR-NEXT:   0x30420 R_AARCH64_RELATIVE -
# RELR-NEXT:   0x30428 R_AARCH64_RELATIVE -
# RELR-NEXT:   0x30452 R_AARCH64_RELATIVE -
# RELR-NEXT: }

# RUN: llvm-readobj -x .test %t2 | FileCheck --check-prefix=HEX %s

# HEX:      Hex dump of section '.test':
# HEX-NEXT: 0x00030410 01000000 2a000020 02000000 2b000000
##                     ^^^^^^^^ Addend = 1
##                              ^^^^ Discr = 42
##                                    ^^ Key (bits 5..6) = DA
##                                       ^^^^^^^^ Addend = 2
##                                                ^^^^ Discr = 43
##                                                      ^^ Key (bits 5..6) = IA
# HEX-NEXT: 0x00030420 03000000 2c000080 78563412 2d000020
##                     ^^^^^^^^ Addend = 3
##                              ^^^^ Discr = 44
##                                    ^^ Key (bits 5..6) = IA
##                                    ^^ Addr diversity (bit 7) = true
##                                       ^^^^^^^^ Addend = 0x12345678
##                                                ^^^^ Discr = 45
##                                                      ^^ Key (bits 5..6) = DA
# HEX-NEXT: 0x00030430 00000000 2e000020 00000000 2f000020
##                     ^^^^^^^^ No implicit addend (rela reloc)
##                              ^^^^ Discr = 46
##                                    ^^ Key (bits 5..6) = DA
##                                       ^^^^^^^^ Addend = 0
##                                                ^^^^ Discr = 47
##                                                      ^^ Key (bits 5..6) = DA
# HEX-NEXT: 0x00030440 00000000 30000000 00000000 00310000
##                     ^^^^^^^^ Addend = 0
##                              ^^^^ Discr = 48
##                                    ^^ Key (bits 5..6) = IA
##                                         ^^^^^^ ^^ No implicit addend (rela reloc)
##                                                  ^^^^ Discr = 49
# HEX-NEXT: 0x00030450 20000500 00003200 0020{{\ }}
##                     ^^ Key (bits 5..6) = DA
##                         ^^^^ ^^^^ Addend = 5
##                                  ^^^^ Discr = 48
##                                         ^^ Key (bits 5..6) = DA

.section .test, "aw"
.p2align 3
.quad (__ehdr_start + 1)@AUTH(da,42)
.quad (__ehdr_start + 2)@AUTH(ia,43)
.quad (__ehdr_start + 3)@AUTH(ia,44,addr)
.quad (__ehdr_start + 0x12345678)@AUTH(da,45)
.quad (__ehdr_start + 0x123456789A)@AUTH(da,46)
.quad zed2@AUTH(da,47)
.quad bar2@AUTH(ia,48)
.byte 00
.quad (__ehdr_start + 4)@AUTH(da,49)
.byte 00
.quad (__ehdr_start + 5)@AUTH(da,50)
