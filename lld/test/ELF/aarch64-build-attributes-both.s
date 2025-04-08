// REQUIRES: aarch64

// Test mc -> little endian, lld -> little endian
// RUN: llvm-mc -triple=aarch64 %s -filetype=obj -o %t.o
// RUN: ld.lld %t.o -o %t.so
// RUN: llvm-readelf -n %t.so | FileCheck %s --check-prefix=NOTE
// Test mc -> little endian, lld -> big endian
// RUN: ld.lld -EB %t.o -o %t.so
// RUN: llvm-readelf -n %t.so | FileCheck %s --check-prefix=NOTE
// Test mc -> big endian, lld -> little endian
// RUN: llvm-mc -triple=aarch64_be %s -filetype=obj -o %t.o
// RUN: ld.lld %t.o -o %t.so
// RUN: llvm-readelf -n %t.so | FileCheck %s --check-prefix=NOTE
// Test mc -> big endian, lld -> big endian
// RUN: llvm-mc -triple=aarch64_be %s -filetype=obj -o %t.o
// RUN: ld.lld -EB %t.o -o %t.so
// RUN: llvm-readelf -n %t.so | FileCheck %s --check-prefix=NOTE


// NOTE: Displaying notes found in: .note.gnu.property
// NOTE-CHECK:   Owner                Data size 	Description
// NOTE-CHECK:  GNU                  0x00000028	NT_GNU_PROPERTY_TYPE_0 (property note)
// NOTE-CHECK:    Properties:    aarch64 feature: PAC
// NOTE-CHECK:        AArch64 PAuth ABI core info: platform 0x12345678 (unknown), version 0x87654321


.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 305419896
.aeabi_attribute Tag_PAuth_Schema, 2271560481
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_PAC, 1

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
