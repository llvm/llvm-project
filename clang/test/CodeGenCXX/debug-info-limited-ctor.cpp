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

// Test for constexpr constructor.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "E"{{.*}}DIFlagTypePassByValue
struct E {
  constexpr E(){};
} TestE;

// Test for trivial constructor.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "F"{{.*}}DIFlagTypePassByValue
struct F {
  F() = default;
  F(int) {}
  int i;
} TestF;

// Test for trivial constructor.
// CHECK-DAG: ![[G:.*]] ={{.*}}!DICompositeType({{.*}}name: "G"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType({{.*}}scope: ![[G]], {{.*}}DIFlagTypePassByValue
struct G {
  G() : g_(0) {}
  struct {
    int g_;
  };
} TestG;

// Test for an aggregate class with an implicit non-trivial default constructor
// that is not instantiated.
// CHECK-DAG: !DICompositeType({{.*}}name: "H",{{.*}}DIFlagTypePassByValue
struct H {
  B b;
};
void f(H h) {}

// Test for an aggregate class with an implicit non-trivial default constructor
// that is instantiated.
// CHECK-DAG: !DICompositeType({{.*}}name: "J",{{.*}}DIFlagTypePassByValue
struct J {
  B b;
};
void f(decltype(J()) j) {}

// Test for a class with trivial default constructor that is not instantiated.
// CHECK-DAG: !DICompositeType({{.*}}name: "K",{{.*}}DIFlagTypePassByValue
class K {
  int i;
};
void f(K k) {}

// Test that we don't use constructor homing on lambdas.
// CHECK-DAG: ![[L:.*]] ={{.*}}!DISubprogram({{.*}}name: "L"
// CHECK-DAG: !DICompositeType({{.*}}scope: ![[L]], {{.*}}DIFlagTypePassByValue
void L() {
  auto func = [&]() {};
}
