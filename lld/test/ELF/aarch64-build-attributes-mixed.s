// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t11.o
// RUN: llvm-mc -triple=aarch64 -filetype=obj merged-mixed-2.s -o %t12.o
// RUN: llvm-mc -triple=aarch64 -filetype=obj merged-mixed-3.s -o %t13.o
// RUN: ld.lld -r %t11.o %t12.o %t13.o -o %t.merged1.o
// RUN: llvm-readelf -n %t.merged1.o | FileCheck %s --check-prefix=NOTE-MIXED

// NOTE-MIXED: Displaying notes found in: .note.gnu.property
// NOTE-MIXED-NEXT:   Owner                Data size 	Description
// NOTE-MIXED-NEXT:   GNU                  0x00000028	NT_GNU_PROPERTY_TYPE_0 (property note)
// NOTE-MIXED-NEXT:     Properties:    aarch64 feature: BTI, PAC
// NOTE-MIXED-NEXT:         AArch64 PAuth ABI core info: platform 0x31 (unknown), version 0x13

/// The Build attributes section appearing in the output of
/// llvm-mc should not appear in the output of lld, because
/// AArch64 build attributes are being transformed into .gnu.properties.

// CHECK: .note.gnu.property
// CHECK-NOT: .ARM.attributes

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1


//--- merged-mixed-2.s
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
  // PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 16          // Data size
  .quad 49          // PAuth ABI platform
  .quad 19          // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:

//--- merged-mixed-3.s
.section .note.gnu.property, "a"
  .align 4
  .long 4                      // namesz
  .long 0x10                   // descsz
  .long 5                      // type (NT_GNU_PROPERTY_TYPE_0)
  .asciz "GNU"                // name (null-terminated)
  .align 4
  .long 0xc0000000             // pr_type (GNU_PROPERTY_AARCH64_FEATURE_1_AND)
  .long 4                      // pr_datasz
  .long 7                      // pr_data: BTI (1), PAC (2), GCS (4) = 0b111 = 7
  .long 0                      // padding or next property
