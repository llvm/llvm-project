// REQUIRES: aarch64

// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.o
// RUN: ld.lld -r %t.o -o %t.invalid.o
// RUN: llvm-readelf -n %t.invalid.o | FileCheck %s

/// According to the BuildAttributes specification Build Attributes
/// A (TagPlatform, TagSchema)of (0, 1) maps to an explicit PAuth property 
/// of platform = 0, version = 0 ('Invalid').

// CHECK:      Displaying notes found in: .note.gnu.property
// CHECK-NEXT:  Owner                Data size 	Description
// CHECK-NEXT:  GNU                  0x00000018	NT_GNU_PROPERTY_TYPE_0 (property note)
// CHECK-NEXT:    Properties:        AArch64 PAuth ABI core info: platform 0x0 (invalid), version 0x0

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 0
.aeabi_attribute Tag_PAuth_Schema, 1
