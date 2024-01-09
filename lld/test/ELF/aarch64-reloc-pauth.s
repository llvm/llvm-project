// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared2.s -o %t.so.o
// RUN: ld.lld -shared %t.so.o -soname=so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld -pie -z nopack-relative-relocs %t.o %t.so -o %t2
// RUN: llvm-readobj -r %t2 | FileCheck --check-prefix=UNPACKED %s

// UNPACKED:          Section ({{.+}}) .rela.dyn {
// UNPACKED-NEXT:       0x303C8 R_AARCH64_AUTH_RELATIVE - 0x1
// UNPACKED-NEXT:       0x303E1 R_AARCH64_AUTH_RELATIVE - 0x2
// UNPACKED-NEXT:       0x303D0 R_AARCH64_AUTH_ABS64 zed2 0x0
// UNPACKED-NEXT:       0x303D8 R_AARCH64_AUTH_ABS64 bar2 0x0
// UNPACKED-NEXT:     }

// RUN: ld.lld -pie -z pack-relative-relocs %t.o %t.so -o %t2
// RUN: llvm-readobj -S --dynamic-table %t2 | FileCheck --check-prefix=RELR-HEADERS %s
// RUN: llvm-readobj -r --raw-relr %t2 | FileCheck --check-prefix=RAW-RELR %s
// RUN: llvm-readobj -r %t2 | FileCheck --check-prefix=RELR %s

// RELR-HEADERS:       Index: 1
// RELR-HEADERS-NEXT:  Name: .dynsym

// RELR-HEADERS:       Name: .relr.auth.dyn
// RELR-HEADERS-NEXT:  Type: SHT_AARCH64_AUTH_RELR
// RELR-HEADERS-NEXT:  Flags [ (0x2)
// RELR-HEADERS-NEXT:    SHF_ALLOC (0x2)
// RELR-HEADERS-NEXT:  ]
// RELR-HEADERS-NEXT:  Address: [[ADDR:.*]]
// RELR-HEADERS-NEXT:  Offset: [[ADDR]]
// RELR-HEADERS-NEXT:  Size: 8
// RELR-HEADERS-NEXT:  Link: 0
// RELR-HEADERS-NEXT:  Info: 0
// RELR-HEADERS-NEXT:  AddressAlignment: 8
// RELR-HEADERS-NEXT:  EntrySize: 8

// RELR-HEADERS:       0x0000000070000012 AARCH64_AUTH_RELR    [[ADDR]]
// RELR-HEADERS:       0x0000000070000011 AARCH64_AUTH_RELRSZ  8 (bytes)
// RELR-HEADERS:       0x0000000070000013 AARCH64_AUTH_RELRENT 8 (bytes)

/// SHT_RELR section contains address/bitmap entries
/// encoding the offsets for relative relocation.
// RAW-RELR:           Section ({{.+}}) .relr.auth.dyn {
// RAW-RELR-NEXT:      0x303E8
// RAW-RELR-NEXT:      }

/// Decoded SHT_RELR section is same as UNPACKED,
/// but contains only the relative relocations.
/// Any relative relocations with odd offset stay in SHT_RELA.

// RELR:      Section ({{.+}}) .rela.dyn {
// RELR-NEXT:   0x30401 R_AARCH64_AUTH_RELATIVE - 0x2
// RELR-NEXT:   0x303F0 R_AARCH64_AUTH_ABS64 zed2 0x0
// RELR-NEXT:   0x303F8 R_AARCH64_AUTH_ABS64 bar2 0x0
// RELR-NEXT: }
// RELR-NEXT: Section ({{.+}}) .relr.auth.dyn {
// RELR-NEXT:   0x303E8 R_AARCH64_RELATIVE -
// RELR-NEXT: }

.section .test, "aw"
.p2align 3
.quad (__ehdr_start + 1)@AUTH(da,42)
.quad zed2@AUTH(da,42)
.quad bar2@AUTH(ia,42)
.byte 00
.quad (__ehdr_start + 2)@AUTH(da,42)
