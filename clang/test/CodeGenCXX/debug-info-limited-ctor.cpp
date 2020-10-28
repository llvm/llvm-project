// RUN: %clang -cc1 -debug-info-kind=constructor -emit-llvm %s -o - | FileCheck %s

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "A"{{.*}}DIFlagTypePassByValue
struct A {
} TestA;

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "B"{{.*}}flags: DIFlagFwdDecl
struct B {
  B();
} TestB;

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "C"{{.*}}DIFlagTypePassByValue
struct C {
  C() {}
} TestC;

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "D"{{.*}}DIFlagTypePassByValue
struct D {
  D();
};
D::D() {}

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "E"{{.*}}DIFlagTypePassByValue
struct E {
  constexpr E(){};
} TestE;

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "F"{{.*}}DIFlagTypePassByValue
struct F {
  F() = default;
  F(int) {}
  int i;
} TestF;
