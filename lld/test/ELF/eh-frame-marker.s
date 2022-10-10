// REQUIRES: x86
/// A symbol can be defined relative to an empty .eh_frame (__EH_FRAME_LIST__).
/// Symbols defined relative to a non-empty .eh_frame have unclear semantics.
/// Test we don't crash.

// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: ld.lld --eh-frame-hdr %t.o -o %t.so -shared
// RUN: llvm-readobj --symbols -S %t.so | FileCheck %s

// CHECK:      Name: .eh_frame_hdr
// CHECK:      Name: .eh_frame
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address:  [[ADDR:.*]]

// CHECK:      Name: foo
// CHECK-NEXT: Value: [[ADDR]]

        .section .eh_frame,"a",@unwind
foo:
        .long 0
