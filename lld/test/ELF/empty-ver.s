// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: ld.lld %t.o -o %t.so -shared -version-script %p/Inputs/empty-ver.ver
// RUN: llvm-readobj -version-info %t.so | FileCheck %s

// CHECK:      Version symbols {
// CHECK-NEXT:   Section Name:
// CHECK-NEXT:   Address:
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Link:
// CHECK-NEXT:   Symbols [
// CHECK-NEXT:     Symbol {
// CHECK-NEXT:       Version: 0
// CHECK-NEXT:       Name: @
// CHECK-NEXT:     }
// CHECK-NEXT:     Symbol {
// CHECK-NEXT:       Version: 2
// CHECK-NEXT:       Name: foo@ver
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }


.global foo@ver
foo@ver:
