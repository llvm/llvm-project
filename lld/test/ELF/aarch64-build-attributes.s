// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t1.o
// RUN: llvm-mc -triple=aarch64 -filetype=obj merged-2.s -o %t2.o
// RUN: llvm-mc -triple=aarch64 -filetype=obj merged-3.s -o %t3.o
// RUN: ld.lld -r %t1.o %t2.o %t3.o -o %t.merged.o
// RUN: llvm-readelf -n %t.merged.o | FileCheck %s --check-prefix=NOTE

// NOTE: Displaying notes found in: .note.gnu.property
// NOTE-NEXT:  Owner                Data size 	Description
// NOTE-NEXT:  GNU                  0x00000028	NT_GNU_PROPERTY_TYPE_0 (property note)
// NOTE-NEXT:    Properties:    aarch64 feature: BTI
// NOTE-NEXT:        AArch64 PAuth ABI core info: platform 0x31 (unknown), version 0x13

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


//--- merged-2.s
.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 0
.aeabi_attribute Tag_Feature_GCS, 1


//--- merged-3.s
.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 0
