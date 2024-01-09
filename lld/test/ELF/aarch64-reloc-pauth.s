// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared2.s -o %t.so.o
// RUN: ld.lld -shared %t.so.o -soname=so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld -pie -z nopack-relative-relocs %t.o %t.so -o %t2
// RUN: llvm-readobj -r %t2 | FileCheck --check-prefix=UNPACKED %s

// UNPACKED:          Section ({{.+}}) .rela.dyn {
// UNPACKED-NEXT:       0x30680 R_AARCH64_AUTH_RELATIVE - 0x1
// UNPACKED-NEXT:       0x30688 R_AARCH64_AUTH_RELATIVE - 0x2
// UNPACKED-NEXT:       0x30690 R_AARCH64_AUTH_RELATIVE - 0x3
// UNPACKED-NEXT:       0x30698 R_AARCH64_AUTH_RELATIVE - 0x4
// UNPACKED-NEXT:       0x306A0 R_AARCH64_AUTH_RELATIVE - 0x5
// UNPACKED-NEXT:       0x306A8 R_AARCH64_AUTH_RELATIVE - 0x6
// UNPACKED-NEXT:       0x306B0 R_AARCH64_AUTH_RELATIVE - 0x7
// UNPACKED-NEXT:       0x306B8 R_AARCH64_AUTH_RELATIVE - 0x8
// UNPACKED-NEXT:       0x306C8 R_AARCH64_AUTH_RELATIVE - 0x1
// UNPACKED-NEXT:       0x306D0 R_AARCH64_AUTH_RELATIVE - 0x2
// UNPACKED-NEXT:       0x306D8 R_AARCH64_AUTH_RELATIVE - 0x3
// UNPACKED-NEXT:       0x306E0 R_AARCH64_AUTH_RELATIVE - 0x4
// UNPACKED-NEXT:       0x306E8 R_AARCH64_AUTH_RELATIVE - 0x5
// UNPACKED-NEXT:       0x306F0 R_AARCH64_AUTH_RELATIVE - 0x6
// UNPACKED-NEXT:       0x306F8 R_AARCH64_AUTH_RELATIVE - 0x7
// UNPACKED-NEXT:       0x30710 R_AARCH64_AUTH_RELATIVE - 0x1
// UNPACKED-NEXT:       0x30718 R_AARCH64_AUTH_RELATIVE - 0x2
// UNPACKED-NEXT:       0x30720 R_AARCH64_AUTH_RELATIVE - 0x3
// UNPACKED-NEXT:       0x30728 R_AARCH64_AUTH_RELATIVE - 0x4
// UNPACKED-NEXT:       0x30730 R_AARCH64_AUTH_RELATIVE - 0x5
// UNPACKED-NEXT:       0x30738 R_AARCH64_AUTH_RELATIVE - 0x6
// UNPACKED-NEXT:       0x30740 R_AARCH64_AUTH_RELATIVE - 0x7
// UNPACKED-NEXT:       0x30748 R_AARCH64_AUTH_RELATIVE - 0x8
// UNPACKED-NEXT:       0x30750 R_AARCH64_AUTH_RELATIVE - 0x9
// UNPACKED-NEXT:       0x30759 R_AARCH64_AUTH_RELATIVE - 0xA
// UNPACKED-NEXT:       0x306C0 R_AARCH64_AUTH_ABS64 bar2 0x1
// UNPACKED-NEXT:       0x30708 R_AARCH64_AUTH_ABS64 bar2 0x0
// UNPACKED-NEXT:       0x30761 R_AARCH64_AUTH_ABS64 bar2 0x0
// UNPACKED-NEXT:       0x30769 R_AARCH64_AUTH_ABS64 bar2 0x0
// UNPACKED-NEXT:       0x30771 R_AARCH64_AUTH_ABS64 bar2 0x1
// UNPACKED-NEXT:       0x30779 R_AARCH64_AUTH_ABS64 bar2 0x1
// UNPACKED-NEXT:       0x30781 R_AARCH64_AUTH_ABS64 bar2 0x0
// UNPACKED-NEXT:       0x30700 R_AARCH64_AUTH_ABS64 zed2 0x0
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
// RELR-HEADERS-NEXT:  Size: 16
// RELR-HEADERS-NEXT:  Link: 0
// RELR-HEADERS-NEXT:  Info: 0
// RELR-HEADERS-NEXT:  AddressAlignment: 8
// RELR-HEADERS-NEXT:  EntrySize: 8

// RELR-HEADERS:       0x0000000070000012 AARCH64_AUTH_RELR    [[ADDR]]
// RELR-HEADERS:       0x0000000070000011 AARCH64_AUTH_RELRSZ  16 (bytes)
// RELR-HEADERS:       0x0000000070000013 AARCH64_AUTH_RELRENT 8 (bytes)

/// SHT_RELR section contains address/bitmap entries
/// encoding the offsets for relative relocation.
// RAW-RELR:           Section ({{.+}}) .relr.auth.dyn {
// RAW-RELR-NEXT:      0x30480
// RAW-RELR-NEXT:      0x7FCFEFF
// RAW-RELR-NEXT:      }

/// Decoded SHT_RELR section is same as UNPACKED,
/// but contains only the relative relocations.
/// Any relative relocations with odd offset stay in SHT_RELA.

// RELR:      Section ({{.+}}) .rela.dyn {
// RELR-NEXT:   0x30559 R_AARCH64_AUTH_RELATIVE - 0xA
// RELR-NEXT:   0x304C0 R_AARCH64_AUTH_ABS64 bar2 0x1
// RELR-NEXT:   0x30508 R_AARCH64_AUTH_ABS64 bar2 0x0
// RELR-NEXT:   0x30561 R_AARCH64_AUTH_ABS64 bar2 0x0
// RELR-NEXT:   0x30569 R_AARCH64_AUTH_ABS64 bar2 0x0
// RELR-NEXT:   0x30571 R_AARCH64_AUTH_ABS64 bar2 0x1
// RELR-NEXT:   0x30579 R_AARCH64_AUTH_ABS64 bar2 0x1
// RELR-NEXT:   0x30581 R_AARCH64_AUTH_ABS64 bar2 0x0
// RELR-NEXT:   0x30500 R_AARCH64_AUTH_ABS64 zed2 0x0
// RELR-NEXT: }
// RELR-NEXT: Section ({{.+}}) .relr.auth.dyn {
// RELR-NEXT:   0x30480 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30488 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30490 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30498 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304A0 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304A8 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304B0 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304B8 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304C8 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304D0 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304D8 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304E0 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304E8 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304F0 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x304F8 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30510 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30518 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30520 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30528 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30530 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30538 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30540 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30548 R_AARCH64_RELATIVE -
// RELR-NEXT:   0x30550 R_AARCH64_RELATIVE -
// RELR-NEXT: }

.section .test, "aw"
.p2align 3
.quad (__ehdr_start + 1)@AUTH(da,42)
.quad (__ehdr_start + 2)@AUTH(da,42)
.quad (__ehdr_start + 3)@AUTH(da,42)
.quad (__ehdr_start + 4)@AUTH(da,42)
.quad (__ehdr_start + 5)@AUTH(da,42)
.quad (__ehdr_start + 6)@AUTH(da,42)
.quad (__ehdr_start + 7)@AUTH(da,42)
.quad (__ehdr_start + 8)@AUTH(da,42)
.quad (bar2 + 1)@AUTH(ia,42)

.quad (__ehdr_start + 1)@AUTH(da,65535)
.quad (__ehdr_start + 2)@AUTH(da,65535)
.quad (__ehdr_start + 3)@AUTH(da,65535)
.quad (__ehdr_start + 4)@AUTH(da,65535)
.quad (__ehdr_start + 5)@AUTH(da,65535)
.quad (__ehdr_start + 6)@AUTH(da,65535)
.quad (__ehdr_start + 7)@AUTH(da,65535)
.quad zed2@AUTH(da,42)
.quad bar2@AUTH(ia,42)

.quad (__ehdr_start + 1)@AUTH(da,0)
.quad (__ehdr_start + 2)@AUTH(da,0)
.quad (__ehdr_start + 3)@AUTH(da,0)
.quad (__ehdr_start + 4)@AUTH(da,0)
.quad (__ehdr_start + 5)@AUTH(da,0)
.quad (__ehdr_start + 6)@AUTH(da,0)
.quad (__ehdr_start + 7)@AUTH(da,0)
.quad (__ehdr_start + 8)@AUTH(da,0)
.quad (__ehdr_start + 9)@AUTH(da,0)
.byte 00
.quad (__ehdr_start + 10)@AUTH(da,0)
.quad bar2@AUTH(ia,42)
.quad bar2@AUTH(ia,42)
.quad (bar2 + 1)@AUTH(ia,42)
.quad (bar2 + 1)@AUTH(ia,42)
.quad bar2@AUTH(ia,42)
