// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/common.s -o %t2
// RUN: ld.lld %t %t2 -o %t3
// RUN: llvm-readobj -t -s %t3 | FileCheck %s
// REQUIRES: x86

// CHECK:      Name: .bss
// CHECK-NEXT: Type: SHT_NOBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x11000
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 22

// CHECK:      Name: sym1
// CHECK-NEXT: Value: 0x11004
// CHECK-NEXT: Size: 8
// CHECK-NEXT: Binding: Global
// CHECK-NEXT: Type: Object
// CHECK-NEXT: Other: 0
// CHECK-NEXT: Section: .bss

// CHECK:      Name: sym2
// CHECK-NEXT: Value: 0x1100C
// CHECK-NEXT: Size: 8
// CHECK-NEXT: Binding: Global
// CHECK-NEXT: Type: Object
// CHECK-NEXT: Other: 0
// CHECK-NEXT: Section: .bss

// CHECK:      Name: sym3
// CHECK-NEXT: Value: 0x11014
// CHECK-NEXT: Size: 2
// CHECK-NEXT: Binding: Global
// CHECK-NEXT: Type: Object
// CHECK-NEXT: Other: 0
// CHECK-NEXT: Section: .bss

// CHECK:      Name: sym4
// CHECK-NEXT: Value: 0x11000
// CHECK-NEXT: Size: 4
// CHECK-NEXT: Binding: Global
// CHECK-NEXT: Type: Object
// CHECK-NEXT: Other: 0
// CHECK-NEXT: Section: .bss


.globl _start
_start:

.comm sym1,4,4
.comm sym2,8,4
.comm sym3,2,2
.comm sym4,4,2
