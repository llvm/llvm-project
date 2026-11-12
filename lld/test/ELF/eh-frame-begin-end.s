// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=amd64-unknown-openbsd %s -o %t.o
// RUN: echo '.section .eh_frame,"a",@unwind; .long 0' | \
// RUN:   llvm-mc -filetype=obj -triple=amd64-unknown-openbsd - -o %t2.o
// RUN: ld.lld %t.o %t2.o -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s

// CHECK:      Name: .eh_frame
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address:
// CHECK-NEXT: Offset: 0x120
// CHECK-NEXT: Size: 4

.section .eh_frame,"a",@unwind
__EH_FRAME_BEGIN__:
