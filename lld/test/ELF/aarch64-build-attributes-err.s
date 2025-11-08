// REQUIRES: aarch64

// RUN: llvm-mc -triple=aarch64 %s -filetype=obj -o %t.o
// RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: GNU properties and build attributes have conflicting AArch64 PAuth data
// ERR-NEXT: GNU properties and build attributes have conflicting AArch64 PAuth data

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 5
.aeabi_attribute Tag_PAuth_Schema, 5
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1

.section ".note.gnu.property", "a"
.long 0x4
.long 0x10
.long 0x5
.asciz "GNU"
.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 0x4
.long 0x2          // GNU_PROPERTY_AARCH64_FEATURE_1_PAC
.long 0x0

.section ".note.gnu.property", "a"
.long 0x4
.long 0x18
.long 0x5
.asciz "GNU"
.long 0xc0000001
.long 0x10
.quad 0x12345678 // platform
.quad 0x87654321  // version
