// REQUIRES: aarch64
// RUN: llvm-mc -triple=aarch64 %s -filetype=obj -o %t.o
// RUN: ld.lld %t.o --shared -o %t.so
// RUN: llvm-readelf --sections %t.so | FileCheck %s
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readelf --sections %t | FileCheck %s
// RUN: ld.lld -r %t.o -o %t2.o
// RUN: llvm-readelf --sections %t2.o | FileCheck %s

/// File has a Build attributes section. This should not appear in
/// ET_EXEC or ET_SHARED files as there is no requirement for it to
/// do so. FIXME, the ld -r (relocatable link) should output a single
/// merged build attributes section. When full support is added in
/// ld.lld this test should be updated.

// CHECK-NOT: .ARM.attributes

.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1

.global _start
.type _start, %function
_start:
ret
