// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: ld.lld -shared %t -o %t2
// RUN: llvm-readobj -t %t2 | FileCheck %s
.long _GLOBAL_OFFSET_TABLE_

// CHECK:      Name: _GLOBAL_OFFSET_TABLE_
// CHECK-NEXT: Value:
// CHECK-NEXT: Size: 0
// CHECK-NEXT: Binding: Local
