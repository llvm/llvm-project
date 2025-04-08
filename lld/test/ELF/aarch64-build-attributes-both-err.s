// REQUIRES: aarch64

// Test mc -> little endian, lld -> little endian
// RUN: llvm-mc -triple=aarch64 %s -filetype=obj -o %t.o
// RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR
// Test mc -> little endian, lld -> big endian
// RUN: llvm-mc -triple=aarch64 %s -filetype=obj -o %t.o
// RUN: not ld.lld -EB %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR
// Test mc -> big endian, lld -> little endian
// RUN: llvm-mc -triple=aarch64_be %s -filetype=obj -o %t.o
// RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR
// Test mc -> big endian, lld -> big endian
// RUN: llvm-mc -triple=aarch64_be %s -filetype=obj -o %t.o
// RUN: not ld.lld -EB %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: Pauth Platform mismatch: file contains both GNU properties and AArch64 build attributes sections
// ERR-NEXT: GNU properties: 305419896
// ERR-NEXT: AArch64 build attributes: 5
// ERR: Pauth Schema mismatch: file contains both GNU properties and AArch64 build attributes sections
// ERR-NEXT: GNU properties: 2271560481
// ERR-NEXT: AArch64 build attributes: 5
// ERR: Features BTI mismatch: file contains both GNU properties and AArch64 build attributes sections
// ERR-NEXT: GNU properties: 0
// ERR-NEXT: AArch64 build attributes: 1
// ERR: Feature GCS mismatch: file contains both GNU properties and AArch64 build attributes sections
// ERR-NEXT: GNU properties: 0
// ERR-NEXT: AArch64 build attributes: 1

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 5
.aeabi_attribute Tag_PAuth_Schema, 5
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1

.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"
.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 2          // GNU_PROPERTY_AARCH64_FEATURE_1_PAC
.long 0

.section ".note.gnu.property", "a"
.long 4
.long 24
.long 5
.asciz "GNU"
.long 0xc0000001
.long 16
.quad 305419896 // platform
.quad 2271560481  // version
