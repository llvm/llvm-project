// REQUIRES: aarch64
// RUN: llvm-mc -triple=aarch64 %s -filetype=obj -o %t.o
// RUN: ld.lld %t.o --shared -o %t.so
// RUN: llvm-readelf --sections %t.so | FileCheck %s
// RUN: llvm-readelf -n %t.so | FileCheck %s --check-prefix=NOTE
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readelf --sections %t | FileCheck %s
// RUN: llvm-readelf -n %t.so | FileCheck %s --check-prefix=NOTE
// RUN: ld.lld -r %t.o -o %t2.o
// RUN: llvm-readelf --sections %t2.o | FileCheck %s
// RUN: llvm-readelf -n %t.so | FileCheck %s --check-prefix=NOTE


/// The Build attributes section appearing in the output of 
/// llvm-mc should not appear in the output of lld, because 
/// AArch64 build attributes are being transformed into .gnu.properties.


// CHECK: .note.gnu.property
// CHECK-NOT: .ARM.attributes

// NOTE: Displaying notes found in: .note.gnu.property
// NOTE-NEXT: Owner Data size Description
// NOTE-NEXT: GNU 0x00000028 NT_GNU_PROPERTY_TYPE_0 (property note)
// NOTE-NEXT: Properties: aarch64 feature: BTI, PAC, GCS
// NOTE-NEXT: AArch64 PAuth ABI core info: platform 0x31 (unknown), version 0x13


.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1

.global _start
.type _start, %function
_start:
ret
