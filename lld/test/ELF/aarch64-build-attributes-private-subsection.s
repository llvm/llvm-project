// REQUIRES: aarch64

/// Test that a build attributes section without 'aeabi_feature_and_bits' and
/// 'aeabi_pauthabi' subsections does not conflict with GNU Program Properties.

// RUN: llvm-mc -triple=aarch64 -mattr=+bti -aarch64-mark-bti-property -filetype=obj %s -o %t.o
// RUN: ld.lld -shared %t.o -z force-bti -o %t.out 2>&1 | count 0
// RUN: llvm-readobj --notes %t.out | FileCheck %s

// RUN: llvm-mc -triple=aarch64 --defsym EMIT_GNU_PROPERTY=1 -filetype=obj %s -o %t.o
// RUN: ld.lld -shared %t.o -z force-bti -o %t.out 2>&1 | count 0
// RUN: llvm-readobj --notes %t.out | FileCheck %s --check-prefixes=CHECK,WITH_PAUTH

// CHECK:      NoteSections [
// CHECK-NEXT:   NoteSection {
// CHECK-NEXT:     Name: .note.gnu.property
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Notes [
// CHECK-NEXT:       {
// CHECK-NEXT:         Owner: GNU
// CHECK-NEXT:         Data size:
// CHECK-NEXT:         Type: NT_GNU_PROPERTY_TYPE_0 (property note)
// CHECK-NEXT:         Property [
// CHECK-NEXT:           aarch64 feature: BTI
// WITH_PAUTH-NEXT:      AArch64 PAuth ABI core info: platform 0x31 (unknown), version 0x13
// CHECK-NEXT:         ]
// CHECK-NEXT:       }
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]

.aeabi_subsection anon_dummy, optional, uleb128
.aeabi_attribute 1, 1

.ifdef EMIT_GNU_PROPERTY
.section ".note.gnu.property", "a"
  .long 0x4           // Name length 4 ("GNU")
  .long end - begin   // Data length
  .long 0x5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"        // Name
  .p2align 3
begin:
  .long 0xc0000000    // GNU_PROPERTY_AARCH64_FEATURE_1_AND
  .long 0x4
  .long 0x1           // GNU_PROPERTY_AARCH64_FEATURE_1_BTI
  .long 0x0
  // PAuth ABI property note
  .long 0xc0000001    // GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 0x10          // Data length
  .quad 0x31          // PAuth ABI platform
  .quad 0x13          // PAuth ABI version
  .p2align 3          // Align to 8 byte for 64 bit
end:
.endif
