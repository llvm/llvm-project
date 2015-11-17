// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld2 %t.o -o %t.so -shared
// RUN: llvm-readobj -s %t.so | FileCheck %s

        .section        .rodata.str1.16,"aMS",@progbits,1
        .align  16
        .asciz "foo"

        .section        .rodata.str1.1,"aMS",@progbits,1
        .asciz "foo"

// CHECK:      Name: .rodata
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_MERGE
// CHECK-NEXT:   SHF_STRINGS
// CHECK-NEXT: ]
// CHECK-NEXT: Address:
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 4
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 16

// CHECK:      Name: .rodata
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_MERGE
// CHECK-NEXT:   SHF_STRINGS
// CHECK-NEXT: ]
// CHECK-NEXT: Address:
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 4
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 1
