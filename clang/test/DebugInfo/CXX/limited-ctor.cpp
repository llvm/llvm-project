// RUN: %clang_cc1 -debug-info-kind=constructor -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=constructor -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM %s

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "A"{{.*}}DIFlagTypePassByValue
struct A {
} TestA;

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "B"{{.*}}flags: DIFlagFwdDecl
struct B {
  B();
} TestB;

// CHECK-DAG: ![[C:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C"{{.*}}DIFlagTypePassByValue
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

// Declared but not defined constexpr constructor should not emit full debug info..
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DeclaredConstexpr"{{.*}}flags: DIFlagFwdDecl
struct DeclaredConstexpr {
  constexpr DeclaredConstexpr();
} TestDeclaredConstexpr;

// Defined out-of-line constexpr constructor should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "OutOfLineConstexpr"{{.*}}DIFlagTypePassByValue
struct OutOfLineConstexpr {
  constexpr OutOfLineConstexpr();
} TestOutOfLineConstexpr;
constexpr OutOfLineConstexpr::OutOfLineConstexpr() {}

// Defined delegating constructor where delegated constructor is not defined
// should not emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Delegating"{{.*}}flags: DIFlagFwdDecl
struct Delegating {
  Delegating() : Delegating(42) {}
  Delegating(int);
} TestDelegating;

// Defined out-of-line delegating constructor where delegated constructor is not
// defined should not emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "OutOfLineDelegating"{{.*}}flags: DIFlagFwdDecl
struct OutOfLineDelegating {
  OutOfLineDelegating();
  OutOfLineDelegating(int);
} TestOutOfLineDelegating;
OutOfLineDelegating::OutOfLineDelegating() : OutOfLineDelegating(42) {}

// Defined delegating constructor where delegated constructor is defined should
// emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingToDefined"{{.*}}DIFlagTypePassByValue
struct DelegatingToDefined {
  DelegatingToDefined() : DelegatingToDefined(42) {}
  DelegatingToDefined(int) {}
} TestDelegatingToDefined;

// Defined delegating constructor where delegated constructor is defined out of
// line should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingToOutOfLine"{{.*}}DIFlagTypePassByValue
struct DelegatingToOutOfLine {
  DelegatingToOutOfLine() : DelegatingToOutOfLine(42) {}
  DelegatingToOutOfLine(int);
} TestDelegatingToOutOfLine;
DelegatingToOutOfLine::DelegatingToOutOfLine(int) {}

// Defined out-of-line delegating constructor where delegated constructor is
// defined should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingOutOfLine"{{.*}}DIFlagTypePassByValue
struct DelegatingOutOfLine {
  DelegatingOutOfLine();
  DelegatingOutOfLine(int) {}
} TestDelegatingOutOfLine;
DelegatingOutOfLine::DelegatingOutOfLine() : DelegatingOutOfLine(42) {}

// Defined out-of-line delegating constructor where delegated constructor is
// defined out-of-line should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingOutOfLineToOutOfLine"{{.*}}DIFlagTypePassByValue
struct DelegatingOutOfLineToOutOfLine {
  DelegatingOutOfLineToOutOfLine();
  DelegatingOutOfLineToOutOfLine(int);
} TestDelegatingOutOfLineToOutOfLine;
DelegatingOutOfLineToOutOfLine::DelegatingOutOfLineToOutOfLine()
    : DelegatingOutOfLineToOutOfLine(42) {}
DelegatingOutOfLineToOutOfLine::DelegatingOutOfLineToOutOfLine(int) {}

// Delegating constructor to a copy constructor should not enable constructor
// homing, so it should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingToCopyCtor"{{.*}}DIFlagTypePassByValue
struct DelegatingToCopyCtor {
  DelegatingToCopyCtor(const DelegatingToCopyCtor&) = default;
  DelegatingToCopyCtor(const DelegatingToCopyCtor& val, int)
      : DelegatingToCopyCtor(val) {}
};
void TestDelegatingToCopyCtor(DelegatingToCopyCtor) {}

// Delegating constructor to a move constructor should not enable constructor
// homing, so it should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingToMoveCtor"{{.*}}DIFlagTypePassByValue
struct DelegatingToMoveCtor {
  DelegatingToMoveCtor(const DelegatingToMoveCtor&) = default;
  DelegatingToMoveCtor(DelegatingToMoveCtor&&) = default;
  DelegatingToMoveCtor(DelegatingToMoveCtor&& val, int)
      : DelegatingToMoveCtor(static_cast<DelegatingToMoveCtor&&>(val)) {}
};
void TestDelegatingToMoveCtor(DelegatingToMoveCtor) {}

// Defined delegating constexpr constructor where delegated constructor is also
// defined should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingConstexpr"{{.*}}DIFlagTypePassByValue
struct DelegatingConstexpr {
  constexpr DelegatingConstexpr() : DelegatingConstexpr(42) {}
  constexpr DelegatingConstexpr(int) {}
} TestDelegatingConstexpr;

// Defined out-of-line delegating constexpr constructor where delegated
// constructor is also defined out-of-line should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingConstexprOutOfLine"{{.*}}DIFlagTypePassByValue
struct DelegatingConstexprOutOfLine {
  constexpr DelegatingConstexprOutOfLine();
  constexpr DelegatingConstexprOutOfLine(int);
} TestDelegatingConstexprOutOfLine;
constexpr DelegatingConstexprOutOfLine::DelegatingConstexprOutOfLine()
    : DelegatingConstexprOutOfLine(42) {}
constexpr DelegatingConstexprOutOfLine::DelegatingConstexprOutOfLine(int) {}

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

// CHECK-DAG: !DICompositeType({{.*}}name: "DeletedCtors",{{.*}}DIFlagTypePassBy
struct NonTrivial {
  NonTrivial();
};
struct DeletedCtors {
  DeletedCtors() = delete;
  constexpr DeletedCtors(int) = delete;
  DeletedCtors(const DeletedCtors &) = default;
  void f1();
  NonTrivial t;
};

const NonTrivial &f(const DeletedCtors &D) {
  return D.t;
}

// Test that we don't use constructor homing on lambdas.
// CHECK-DAG: ![[L:.*]] ={{.*}}!DISubprogram({{.*}}name: "L"
// CHECK-DAG: !DICompositeType({{.*}}scope: ![[L]], {{.*}}DIFlagTypePassByValue
void L() {
  auto func = [&]() {};
}

// Check that types are being added to retained types list.
// CHECK-DAG: !DICompileUnit{{.*}}retainedTypes: ![[RETAINED:[0-9]+]]
// CHECK-DAG: ![[RETAINED]] = {{.*}}![[C]]


struct VTableAndCtor {
  virtual void f1();
  VTableAndCtor();
};

VTableAndCtor::VTableAndCtor() {
}

// ITANIUM-DAG: !DICompositeType({{.*}}name: "VTableAndCtor", {{.*}}flags: DIFlagFwdDecl

