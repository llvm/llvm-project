// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple i686-pc-windows-msvc -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=MSVC

template<typename T> struct Identity {
  typedef T Type;
};

void f(Identity<int>::Type a) {}
void f(Identity<int> a) {}
void f(int& a) { }

template<typename T> struct A {
  A<T> *next;
};
void f(A<int>) { }

struct B { };

void f() {
  int B::*a = 0;
  void (B::*b)() = 0;
}

namespace EmptyNameCrash {
  struct A { A(); };
  typedef struct { A x; } B;
  B x;
}

// PR4890
namespace PR4890 {
  struct X {
    ~X();
  };

  X::~X() { }
}

namespace VirtualDtor {
  struct Y {
    virtual ~Y();
  };
  
  Y::~Y() { }
}

namespace VirtualBase {
  struct A { int a; };
  struct B : virtual A { int b; };

  void f() {
    B b;
  }
}

// CHECK: define void @_ZN7pr147634funcENS_3fooE
// CHECK: call void @llvm.dbg.declare({{.*}}, metadata ![[F:.*]], metadata ![[EXPR:.*]])

// MSVC: [[VBASE_B:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B",{{.*}} line: 49
// MSVC-SAME:                                            size: 96, align: 32
// MSVC-NOT:                                             offset:
// MSVC-NOT:                                             DIFlagFwdDecl
// MSVC-SAME:                                            elements: [[VBASE_B_DEF:![0-9]+]]
// MSVC: [[VBASE_B_DEF]] = !{[[VBASE_A_IN_B:![0-9]+]],
//
// Look for the vbtable offset of A, which should be 4.
// MSVC: [[VBASE_A_IN_B]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: [[VBASE_B]],
// MSVC-SAME:                              baseType: !{{[0-9]*}}

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "B",{{.*}} line: 49,
// CHECK-SAME:             size: 128, align: 64,
// CHECK-NOT:              offset:
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             elements: [[VBASE_B_DEF:![^,)]+]]
// CHECK: [[VBASE_B_DEF]] = !{[[VBASE_A_IN_B:![0-9]+]],
//
// Look for the vtable offset offset, which should be -24.
// CHECK: [[VBASE_A_IN_B]] = !DIDerivedType(tag: DW_TAG_inheritance
// CHECK-SAME:                              scope: !"_ZTSN11VirtualBase1BE"
// CHECK-SAME:                              baseType: !"_ZTSN11VirtualBase1AE"
// CHECK-SAME:                              offset: 24,
namespace b5249287 {
template <typename T> class A {
  struct B;
};

class Cls {
  template <typename T> friend class A<T>::B;
};

Cls obj;
}

namespace pr14763 {
struct foo {
  foo(const foo&);
};

foo func(foo f) {
  return f; // reference 'f' for now because otherwise we hit another bug
}

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK-SAME:             scope: [[PR14763:![0-9]+]]
// CHECK-SAME:             identifier: "[[FOO:.*]]"
// CHECK: [[PR14763]] = !DINamespace(name: "pr14763"
// CHECK: [[INCTYPE:![0-9]*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "incomplete"
// CHECK-SAME:                                   DIFlagFwdDecl
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "a"
// CHECK-SAME:             elements: [[A_MEM:![0-9]+]]
// CHECK-SAME:             identifier: "_ZTSN7pr162141aE"
// CHECK: [[A_MEM]] = !{[[A_I:![0-9]*]]}
// CHECK: [[A_I]] = !DIDerivedType(tag: DW_TAG_member, name: "i"
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "b"
// CHECK-SAME:             DIFlagFwdDecl

// CHECK: [[FUNC:![0-9]+]] = distinct !DISubprogram(name: "func", linkageName: "_ZN7pr147634funcENS_3fooE"
// CHECK-SAME:                                      type: [[FUNC_TYPE:![0-9]*]]
// CHECK-SAME:                                      isDefinition: true
}

void foo() {
  const wchar_t c = L'x';
  wchar_t d = c;
}

// CHECK-NOT: !DIGlobalVariable(name: "c"

namespace pr9608 { // also pr9600
struct incomplete;
incomplete (*x)[3];
// CHECK: !DIGlobalVariable(name: "x", linkageName: "_ZN6pr96081xE"
// CHECK-SAME:              type: [[INCARRAYPTR:![0-9]*]]
// CHECK-SAME:              variable: [3 x i8]** @_ZN6pr96081xE
// CHECK: [[INCARRAYPTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[INCARRAY:![0-9]+]]
// CHECK: [[INCARRAY]] = !DICompositeType(tag: DW_TAG_array_type
// CHECK-NOT:                             line:
// CHECK-NOT:                             size:
// CHECK-NOT:                             align:
// CHECK-NOT:                             offset:
// CHECK-SAME:                            baseType: !"_ZTSN6pr960810incompleteE"
}

// For some reason function arguments ended up down here
// CHECK: ![[F]] = !DILocalVariable(name: "f", arg: 1, scope: [[FUNC]]
// CHECK-SAME:                      type: !"[[FOO]]"
// CHECK: ![[EXPR]] = !DIExpression(DW_OP_deref)

// CHECK: !DILocalVariable(name: "c"
// CHECK-NOT:              arg:
// CHECK-SAME:            )

namespace pr16214 {
struct a {
  int i;
};

typedef a at;

struct b {
};

typedef b bt;

void func() {
  at a_inst;
  bt *b_ptr_inst;
  const bt *b_cnst_ptr_inst;
}

}
