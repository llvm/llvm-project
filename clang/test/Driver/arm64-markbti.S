// REQUIRES: aarch64-registered-target

// When -mmark-bti-property is passed the generated file object gets BTI marking.
// RUN: %clang -target arm64-linux-none -mmark-bti-property -c -o - %s | llvm-readobj -n - | FileCheck -check-prefix=CHECK  -check-prefix=CHECK_GEN %s
// RUN: %clang -target arm64-linux-none -DNOTE_PRESENT -c %s -o - | llvm-readobj -n - | FileCheck -check-prefix=CHECK  -check-prefix=CHECK_PRESET %s
// RUN: %clang -target arm64-linux-none -mmark-bti-property -DNOTE_PRESENT -c %s -o - | llvm-readobj -n - | FileCheck -check-prefix=CHECK  -check-prefix=CHECK_PRESET %s
// RUN: %clang -target arm64-linux-none -mmark-bti-property -DNOTE_PRESENT -c %s -o - 2>&1 |  FileCheck -check-prefix=CHECK_WARNING %s
//
// CHECK_WARNING: The .note.gnu.property is not emitted because it is already present.
// CHECK: Name: .note.gnu.property
// CHECK: Type: NT_GNU_PROPERTY_TYPE_0
// CHECK_GEN: aarch64 feature: BTI
// CHECK_PRESET: aarch64 feature: BTI, PAC

#ifdef NOTE_PRESENT
  .section .note.gnu.property, "a";
  .balign 8;
  .long 4;
  .long 0x10;
  .long 0x5
  .asciz "GNU"
  .long 0xc0000000
  .long 4
  .long 3
  .long 0
#endif
