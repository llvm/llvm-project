# REQUIRES: aarch64

# RUN: rm -rf %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared2.s -o %t.a.o
# RUN: ld.lld -shared %t.a.o -soname=so -o %t.a.so
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o

# RUN: ld.lld -pie %t.o %t.a.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=UNPACKED %s

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

# RUN: ld.lld %t.o %t.a.so -o %t.nopie
# RUN: llvm-readobj -r %t.nopie | FileCheck --check-prefix=NOPIE %s

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

# RUN: ld.lld -pie %t.o %t.a.so -o %t.pie
# RUN: llvm-readelf -S -d -r -x .test %t.pie | FileCheck --check-prefixes=PIE,HEX %s

# PIE:      Section Headers:
# PIE-NEXT: Name Type Address Off Size ES Flg Lk Inf Al
# PIE:      .rela.dyn RELA {{0*}}[[#%x,ADDR1:]]
# PIE-SAME:                                     {{0*}}[[#ADDR1]] 000108 18 A 1 0 8

# PIE:      Relocation section '.rela.dyn' at offset 0x[[#ADDR1]] contains 11 entries:
# PIE-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# PIE-NEXT: 0000000000030470  0000000000000411 R_AARCH64_AUTH_RELATIVE 1
# PIE-NEXT: 0000000000030478  0000000000000411 R_AARCH64_AUTH_RELATIVE 30472
# PIE-NEXT: 0000000000030480  0000000000000411 R_AARCH64_AUTH_RELATIVE fffffffffffffffd
# PIE-NEXT: 0000000000030488  0000000000000411 R_AARCH64_AUTH_RELATIVE 12345678
# PIE-NEXT: 0000000000030490  0000000000000411 R_AARCH64_AUTH_RELATIVE 123456789a
# PIE-NEXT: 0000000000030498  0000000000000411 R_AARCH64_AUTH_RELATIVE ffffffedcba98766
# PIE-NEXT: 00000000000304a0  0000000000000411 R_AARCH64_AUTH_RELATIVE 8003046f
# PIE-NEXT: 00000000000304b9  0000000000000411 R_AARCH64_AUTH_RELATIVE 4
# PIE-NEXT: 00000000000304c2  0000000000000411 R_AARCH64_AUTH_RELATIVE 30475
# PIE-NEXT: 00000000000304a8  0000000100000244 R_AARCH64_AUTH_ABS64   0000000000000000 zed2 + 1111
# PIE-NEXT: 00000000000304b0  0000000200000244 R_AARCH64_AUTH_ABS64   0000000000000000 bar2 + 0

# HEX:      Hex dump of section '.test':
# HEX-NEXT: 0x00030470 00000000 2a000020 00000000 2b000000
##                              ^^^^ Discr = 42
##                                    ^^ Key (bits 5..6) = DA
##                                                ^^^^ Discr = 43
##                                                      ^^ Key (bits 5..6) = IA
# HEX-NEXT: 0x00030480 00000000 2c000080 00000000 2d000020
##                              ^^^^ Discr = 44
##                                    ^^ Key (bits 5..6) = IA
##                                    ^^ Addr diversity (bit 7) = true
##                                                ^^^^ Discr = 45
##                                                      ^^ Key (bits 5..6) = DA
# HEX-NEXT: 0x00030490 00000000 2e000020 00000000 2f000020
##                              ^^^^ Discr = 46
##                                    ^^ Key (bits 5..6) = DA
##                                                ^^^^ Discr = 47
##                                                      ^^ Key (bits 5..6) = DA
# HEX-NEXT: 0x000304a0 00000000 30000020 00000000 31000020
##                              ^^^^ Discr = 48
##                                    ^^ Key (bits 5..6) = DA
##                                                ^^^^ Discr = 49
##                                                      ^^ Key (bits 5..6) = DA
# HEX-NEXT: 0x000304b0 00000000 32000000 77000000 00330000
##                              ^^^^ Discr = 50
##                                    ^^ Key (bits 5..6) = IA
##                                                  ^^^^ Discr = 51
# HEX-NEXT: 0x000304c0 20770000 00003400 0020{{\ }}
##                     ^^ Key (bits 5..6) = DA
##                                  ^^^^ Discr = 52
##                                         ^^ Key (bits 5..6) = DA

.section .test, "aw"
.p2align 3
.quad (__ehdr_start + 1)@AUTH(da,42)
.quad (.test + 2)@AUTH(ia,43)
.quad (__ehdr_start - 3)@AUTH(ia,44,addr)
.quad (__ehdr_start + 0x12345678)@AUTH(da,45)
.quad (__ehdr_start + 0x123456789A)@AUTH(da,46)
.quad (__ehdr_start - 0x123456789A)@AUTH(da,47)
.quad (.test + 0x7FFFFFFF)@AUTH(da,48)
.quad (zed2 + 0x1111)@AUTH(da,49)
.quad bar2@AUTH(ia,50)
.byte 0x77
.quad (__ehdr_start + 4)@AUTH(da,51)
.byte 0x77
.quad (.test + 5)@AUTH(da,52)
