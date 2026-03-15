# REQUIRES: aarch64

# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.o
# RUN: ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: (.ARM.attributes): unexpected end of data at offset 0x3f while reading [0x3d, 0x41)

.section .ARM.attributes,"",%0x70000003
.byte 0x41                               // Tag 'A' (format version)
.long 0x00000019                         // Subsection length
.asciz "aeabi_pauthabi"                  // Subsection name
.byte 0x00, 0x00                         // Optionality and Type
.byte 0x01, 0x01, 0x02, 0x01             // PAuth_Platform and PAuth_Schema
.long 0x00000023                         // Subsection length
.asciz "aeabi_feature_and_bits"          // Subsection name
.byte 0x01, 0x00                         // Optionality and Type
.byte 0x00, 0x01, 0x01, 0x01, 0x02, 0x01 // BTI, PAC, GCS
.byte 0x00, 0x00                         // This is the malformation, data is too long.
