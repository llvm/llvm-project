// REQUIRES: aarch64

// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t11.o
// RUN: llvm-mc -triple=aarch64 -filetype=obj merged-property.s -o %t12.o
// RUN: llvm-mc -triple=aarch64 -filetype=obj merged-property2.s -o %t13.o
// RUN: ld.lld -r %t11.o %t12.o %t13.o -o %t.merged1.o
// RUN: llvm-readelf -n %t.merged1.o | FileCheck %s --check-prefix=NOTE-MIXED

/// This test verifies merging of AArch64 build attributes and GNU property notes.
/// Three object files are combined: one with build attributes (PAuth information, BTI, PAC, GCS),
/// and two with GNU property notes encoding the same feature bits. 
/// PAuth ABI info is provided in one of the files and it is expected to be preserved in the merged output.

// NOTE-MIXED: Displaying notes found in: .note.gnu.property
// NOTE-MIXED-NEXT:   Owner                Data size 	Description
// NOTE-MIXED-NEXT:   GNU                  0x00000028	NT_GNU_PROPERTY_TYPE_0 (property note)
// NOTE-MIXED-NEXT:     Properties:    aarch64 feature: BTI, PAC
// NOTE-MIXED-NEXT:         AArch64 PAuth ABI core info: platform 0x31 (unknown), version 0x13

// CHECK: .note.gnu.property
// CHECK-NOT: .ARM.attributes

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1


//--- merged-property.s
.section ".note.gnu.property", "a"
  .long 0x4           // Name length is always 4 ("GNU")
  .long end - begin   // Data length
  .long 0x5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"        // Name
  .p2align 0x3
begin:
  .long 0xc0000000    // GNU_PROPERTY_AARCH64_FEATURE_1_AND
  .long 0x4
  .long 0x7           // pr_data: BTI (1), PAC (2), GCS (4) = 0b111 = 7
  .long 0x0
  // PAuth ABI property note
  .long 0xc0000001    // GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 0x10          // Data length
  .quad 0x31          // PAuth ABI platform
  .quad 0x13          // PAuth ABI version
  .p2align 0x3        // Align to 8 byte for 64 bit
end:

//--- merged-property2.s
.section .note.gnu.property, "a"
  .align 0x4
  .long 0x4           // Name length is always 4 ("GNU")
  .long end2 - begin2 // Data length
  .long 0x5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"        // Name
begin2:
  .align 0x4
  .long 0xc0000000    // Type: GNU_PROPERTY_AARCH64_FEATURE_1_AND
  .long 0x4           // Data length
  .long 0x7           // pr_data: BTI (1), PAC (2), GCS (4) = 0b111 = 7
  .long 0x0
end2:
