// RUN: %clang_cc1 -emit-interface-stubs -fblocks -o - %s | FileCheck %s

// CHECK: --- !ifs-v1
// CHECK-NEXT: IfsVersion: 3.0
// CHECK-NEXT: Target:
// CHECK-NEXT: Symbols:
// CHECK-NEXT: ...
static void (^f)(void*) = ^(void* data) { int i; };
