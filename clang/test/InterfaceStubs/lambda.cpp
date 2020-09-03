// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-interface-stubs -o - %s \
// RUN:     | FileCheck %s

// CHECK: --- !experimental-ifs-v2
// CHECK-NEXT: IfsVersion: 2.0
// CHECK-NEXT: Triple:
// CHECK-NEXT: ObjectFileFormat: ELF
// CHECK-NEXT: Symbols:
// CHECK-NEXT:   f", Type: Object, Size: 1 }
// CHECK-NEXT: ...
auto f = [](void* data) { int i; };
