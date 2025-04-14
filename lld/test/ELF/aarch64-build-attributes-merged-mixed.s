# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t1.o
# RUN: llvm-mc -triple=aarch64 -filetype=obj merged-mixed-2.s -o %t2.o
# RUN: llvm-mc -triple=aarch64 -filetype=obj merged-mixed-3.s -o %t3.o
# RUN: ld.lld -r %t1.o %t2.o %t3.o -o %t.merged.o
# RUN: llvm-readelf -n %t.merged.o | FileCheck %s

# CHECK: Displaying notes found in: .note.gnu.property
# CHECK-NEXT:   Owner                Data size 	Description
# CHECK-NEXT:   GNU                  0x00000028	NT_GNU_PROPERTY_TYPE_0 (property note)
# CHECK-NEXT:     Properties:    aarch64 feature: BTI, PAC
# CHECK-NEXT:         AArch64 PAuth ABI core info: platform 0x31 (unknown), version 0x13


.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1

#--- merged-mixed-2.s
.section ".note.gnu.property", "a"
  .long 4           // Name length is always 4 ("GNU")
  .long end - begin // Data length
  .long 5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"      // Name
  .p2align 3
begin:
  .long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
  .long 4
  .long 7          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI, PAC and GCS
  .long 0
  # PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 16          // Data size
  .quad 49          // PAuth ABI platform
  .quad 19          // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:

#--- merged-mixed-3.s
.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"
.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 3          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI and PAC
.long 0
