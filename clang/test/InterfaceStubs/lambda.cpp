// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-interface-stubs -o - %s \
// RUN:     | FileCheck %s

// CHECK: --- !ifs-v1
// CHECK-NEXT: IfsVersion: 3.0
// CHECK-NEXT: Target:
// CHECK-NEXT: Symbols:
// CHECK-NEXT:   f", Type: Object, Size: 1 }
// CHECK-NEXT: ...
auto f = [](void* data) { int i; };
