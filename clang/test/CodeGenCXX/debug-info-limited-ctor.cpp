// RUN: %clang -cc1 -debug-info-kind=constructor -emit-llvm %s -o - | FileCheck %s

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "A"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
struct A {};
void TestA() { A a; }

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "B"
// CHECK-SAME:             flags: DIFlagFwdDecl
struct B {
  B();
};
void TestB() { B b; }

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "C"
// CHECK-NOT:              flags: DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
struct C {
  C() {}
};
void TestC() { C c; }

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "D"
// CHECK-NOT:              flags: DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
struct D {
  D();
};
D::D() {}
