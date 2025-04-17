# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.o
# RUN: ld.lld %t.o /dev/null 2>&1 | FileCheck %s

# CHECK: (.ARM.attributes): invalid Extended Build Attributes subsection size at offset: 39

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
