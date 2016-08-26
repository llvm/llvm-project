// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: ld.lld --eh-frame-hdr %t.o -o %t.so -shared
// RUN: llvm-readobj -t -s %t.so | FileCheck %s
// We used to crash on this.

// CHECK:      Name: .eh_frame
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x200

// CHECK:      Name: foo
// CHECK-NEXT: Value: 0x200

        .section .eh_frame
foo:
        .long 0
