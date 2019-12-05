// RUN: %clang_cc1 -o - -emit-interface-stubs %s | FileCheck %s

// CHECK:      --- !experimental-ifs-v1
// CHECK-NEXT: IfsVersion: 1.0
// CHECK-NEXT: Triple:
// CHECK-NEXT: ObjectFileFormat: ELF
// CHECK-NEXT: Symbols:
// CHECK-NEXT: ...

 // ConstructorUsingShadowDecl
struct Base { Base(int); };
struct Derived : public Base { using Base::Base; };
