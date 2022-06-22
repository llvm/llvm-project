struct foo;
void func(foo *f) {
}
class bar;
void func(bar *f) {
}
union baz;
void func(baz *f) {
}

class B {
public:
  virtual ~B();
};

B::~B() { extern void mayThrow(); mayThrow();
}

struct C {
  static int s;
  virtual ~C();
};

C::~C() {
}

struct D {
  D();
  virtual ~D();
  void func() {
  }
};

struct E {
  virtual ~E();
  virtual void func() {
  }
};

struct F {
  struct F_inner {
  };
  static const int i = 2;
  virtual ~F();
};

struct G {
  virtual void func();
  struct G_inner {
    int j;
  };
};

struct H {};
struct I : virtual H {};
struct J : I {};
J j;

struct K {
  virtual void func() {
  }
};

struct A {
  int one;
  static const int HdrSize = 52;
  int two;
  A() {
    int x = 1;
  }
};

namespace {
struct L {
  void func() {
  }
};
}

void f1() {
  D x;
  x.func();
  E y;
  int i = F::i;
  F::F_inner z;
  K k;
  k.func();
  L l;
  l.func();
}

int main(int argc, char **argv) {
  B b;
  G::G_inner c_i;
  if (argc) {
    A a;
  }
  return 0;
}

// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown_unknown -emit-llvm -debug-info-kind=limited -fexceptions -std=c++98 %s -o - | FileCheck -check-prefix=CHECK98 -check-prefix=CHECK %s
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-cygwin -emit-llvm -debug-info-kind=limited -fexceptions -std=c++98 %s -o - | FileCheck -check-prefix=CHECK98 -check-prefix=CHECK %s
// RUN: %clang_cc1 -no-opaque-pointers -triple armv7l-unknown-linux-gnueabihf -emit-llvm -debug-info-kind=limited -fexceptions -std=c++98 %s -o - | FileCheck -check-prefix=CHECK98 -check-prefix=CHECK %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown_unknown -emit-llvm -debug-info-kind=limited -fexceptions -std=c++11 %s -o - | FileCheck -check-prefix=CHECK11 -check-prefix=CHECK %s
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-cygwin -emit-llvm -debug-info-kind=limited -fexceptions -std=c++11 %s -o - | FileCheck -check-prefix=CHECK11 -check-prefix=CHECK %s
// RUN: %clang_cc1 -no-opaque-pointers -triple armv7l-unknown-linux-gnueabihf -emit-llvm -debug-info-kind=limited -fexceptions -std=c++11 %s -o - | FileCheck -check-prefix=CHECK11 -check-prefix=CHECK %s

// CHECK98: invoke {{.+}} @_ZN1BD1Ev(%class.B* {{[^,]*}} %b)
// CHECK98-NEXT: unwind label %{{.+}}, !dbg ![[EXCEPTLOC:.*]]
// CHECK11: call {{.+}} @_ZN1BD1Ev(%class.B* {{[^,]*}} %b){{.*}}, !dbg ![[EXCEPTLOC:.*]]

// CHECK: store i32 0, i32* %{{.+}}, !dbg ![[RETLOC:.*]]

// CHECK: [[F:![0-9]*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "F"
// CHECK-SAME:                             DIFlagFwdDecl
// CHECK-NOT:                             identifier:
// CHECK-SAME:                            ){{$}}
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "A"
// CHECK: ![[INT:[0-9]+]] = !DIBasicType(name: "int"
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "HdrSize"
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "I"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "bar"
// CHECK: !DICompositeType(tag: DW_TAG_union_type, name: "baz"
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "B"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "_vptr$B",
// CHECK-SAME:           DIFlagArtificial

// CHECK: [[C:![0-9]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C",
// CHECK-NOT:                              DIFlagFwdDecl
// CHECK-SAME:                             elements: [[C_MEM:![0-9]*]]
// CHECK-SAME:                             vtableHolder: [[C]]
// CHECK-NOT:                              identifier:
// CHECK-SAME:                             ){{$}}
// CHECK: [[C_MEM]] = !{[[C_VPTR:![0-9]*]], [[C_S:![0-9]*]], [[C_DTOR:![0-9]*]]}
// CHECK: [[C_VPTR]] = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$C"
// CHECK-SAME:                        DIFlagArtificial
// CHECK: [[C_S]] = !DIDerivedType(tag: DW_TAG_member, name: "s"
// CHECK-SAME:                     baseType: ![[INT]]
// CHECK-SAME:                     DIFlagStaticMember
// CHECK: [[C_DTOR]] = !DISubprogram(name: "~C"

// CHECK: [[D:![0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "D"
// CHECK-SAME:             size:
// CHECK-SAME:             DIFlagFwdDecl
// CHECK-NOT:              identifier:
// CHECK-SAME:             ){{$}}

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "E"
// CHECK-SAME:             DIFlagFwdDecl
// CHECK-NOT:              identifier:
// CHECK-SAME:             ){{$}}

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "K"
// CHECK-SAME:             identifier: "_ZTS1K"
// CHECK-SAME:             ){{$}}

// CHECK: [[L:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "L"
// CHECK-SAME:             ){{$}}
// CHECK: [[L_FUNC_DECL:![0-9]*]] = !DISubprogram(name: "func",{{.*}} scope: [[L]]
// CHECK-SAME:             DISPFlagLocalToUnit

// CHECK: !DISubprogram(name: "func",{{.*}} scope: [[D]]
// CHECK-SAME:          DISPFlagDefinition
// CHECK-SAME:          declaration: [[D_FUNC_DECL:![0-9]*]]
// CHECK: [[D_FUNC_DECL]] = !DISubprogram(name: "func",{{.*}} scope: [[D]]

// CHECK: !DISubprogram(name: "func",{{.*}} scope: [[L]]
// CHECK-SAME:          DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-SAME:          declaration: [[L_FUNC_DECL]]

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "G_inner",
// CHECK-NOT: DIFlagFwdDecl
// CHECK-SAME: elements: [[G_INNER_MEM:![0-9]*]]
// CHECK-SAME: identifier: "_ZTSN1G7G_innerE"

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "G"
// CHECK-SAME:             DIFlagFwdDecl
// CHECK-NOT:              identifier:
// CHECK-SAME:             ){{$}}
// CHECK: [[G_INNER_MEM]] = !{[[G_INNER_I:![0-9]*]]}
// CHECK: [[G_INNER_I]] = !DIDerivedType(tag: DW_TAG_member, name: "j"
// CHECK-SAME:                           baseType: ![[INT]]

//
// CHECK: ![[EXCEPTLOC]] = !DILocation(line: 99,
// CHECK: ![[RETLOC]] = !DILocation(line: 98,
