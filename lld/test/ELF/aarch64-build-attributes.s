// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o 1.o
// RUN: llvm-mc -triple=aarch64 -filetype=obj pauth-bti-gcs.s -o 2.o
// RUN: llvm-mc -triple=aarch64 -filetype=obj pauth-bti-pac.s -o 3.o
// RUN: ld.lld -r 1.o 2.o 3.o -o merged.o
// RUN: llvm-readelf -n merged.o | FileCheck %s --check-prefix=NOTE

/// This test merges three object files with AArch64 build attributes.
/// All contain identical PAuth ABI info (platform/version), which must be preserved.
/// Only BTI is common across all three in the AND feature set, so the merged output
/// must show BTI only. PAC and GCS are present in subsets and should not appear.

// NOTE: Displaying notes found in: .note.gnu.property
// NOTE-NEXT:  Owner                Data size 	Description
// NOTE-NEXT:  GNU                  0x00000028	NT_GNU_PROPERTY_TYPE_0 (property note)
// NOTE-NEXT:    Properties:    aarch64 feature: BTI
// NOTE-NEXT:        AArch64 PAuth ABI core info: platform 0x31 (unknown), version 0x13

// CHECK: .note.gnu.property
// CHECK-NOT: .ARM.attributes

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1


//--- pauth-bti-gcs.s
.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 0
.aeabi_attribute Tag_Feature_GCS, 1


//--- pauth-bti-pac.s
.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 0
