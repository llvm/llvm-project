// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld %t.o %S/Inputs/segment-start.script -shared -o %t.so
// RUN: llvm-readobj --dyn-symbols %t.so | FileCheck %s

// CHECK:      Name: foobar1
// CHECK-NEXT: Value: 0x8001

// CHECK:      Name: foobar2
// CHECK-NEXT: Value: 0x8002

// CHECK:      Name: foobar3
// CHECK-NEXT: Value: 0x8003

// CHECK:      Name: foobar
// CHECK-NEXT: Value: 0x8004

.data
.quad foobar1
.quad foobar2
.quad foobar3
.quad foobar4
