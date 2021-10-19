// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=BOTH
// RUN: %clang_cc1 -triple i686-pc-windows-msvc -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=MSVC --check-prefix=BOTH

// CHECK: @_ZN6pr96081xE ={{.*}} global [3 x i8]* null, align 8, !dbg [[X:![0-9]+]]

// CHECK: define{{.*}} void @_ZN7pr147634funcENS_3fooE
// CHECK: call void @llvm.dbg.declare({{.*}}, metadata ![[F:[0-9]+]], metadata !DIExpression())

// !llvm.dbg.cu pulls in globals and their types first.
// CHECK-NOT: !DIGlobalVariable(name: "c"
// CHECK: [[X]] = !DIGlobalVariableExpression(var: [[XV:!.*]], expr: !DIExpression())
// CHECK: [[XV]] = distinct !DIGlobalVariable(name: "x", linkageName: "_ZN6pr96081xE"
// CHECK-SAME:                                type: [[INCARRAYPTR:![0-9]*]]
// CHECK: [[INCARRAYPTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[INCARRAY:![0-9]+]]
// CHECK: [[INCARRAY]] = !DICompositeType(tag: DW_TAG_array_type
// CHECK-NOT:                             line:
// CHECK-NOT:                             size:
// CHECK-NOT:                             align:
// CHECK-NOT:                             offset:
// CHECK-SAME:                            baseType: ![[INCTYPE:[0-9]+]]

// CHECK: ![[INCTYPE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "incomplete"
// CHECK-SAME:                                   DIFlagFwdDecl

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
// BOTH: ![[VBASE_B:[0-9]+]] ={{.*}}!DICompositeType(tag: DW_TAG_structure_type, name: "B",{{.*}} line: [[@LINE-1]],
// MSVC-SAME:                                        size: 96
// CHECK-SAME:                                       size: 128,
// BOTH-NOT:                                         offset:
// BOTH-NOT:                                         DIFlagFwdDecl
// BOTH-SAME:                                        elements: [[VBASE_B_DEF:![0-9]+]]
// BOTH: [[VBASE_B_DEF]] = !{[[VBASE_A_IN_B:![0-9]+]],
//
// Look for the vbtable offset of A, which should be 4 for MSVC, 24 otherwise.
// BOTH: [[VBASE_A_IN_B]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[VBASE_B]],
// BOTH-SAME:                              baseType: ![[VBASE_A:[0-9]+]],
// MSVC-SAME:                              offset: 4,
// CHECK-SAME:                             offset: 24,
//
// BOTH: ![[VBASE_A]] ={{.*}}!DICompositeType(tag: DW_TAG_structure_type, name: "A",

  void f() {
    B b;
  }
}

namespace b5249287 {
template <typename T> class A {
  struct B;
};

class Cls {
  template <typename T> friend class A<T>::B;
};

Cls obj;
}

// CHECK: [[FUNC:[0-9]+]] = distinct !DISubprogram(name: "func", linkageName: "_ZN7pr147634funcENS_3fooE"
// CHECK-SAME:                                      type: {{![0-9]+}}
// CHECK-SAME:                                      DISPFlagDefinition

// CHECK: [[PR14763:![0-9]+]] = !DINamespace(name: "pr14763"
namespace pr14763 {
struct foo {
// CHECK: ![[FOO:[0-9]+]] ={{.*}}!DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK-SAME:             scope: [[PR14763]]
// CHECK-SAME:             identifier:
  foo(const foo&);
};

// For some reason function arguments ended up down here
// CHECK: ![[F]] = !DILocalVariable(name: "f", arg: 1, scope: ![[FUNC]]
// CHECK-SAME:                      type: ![[FOO]]
foo func(foo f) {
  return f; // reference 'f' for now because otherwise we hit another bug
}

}

void foo() {
// CHECK: !DILocalVariable(name: "c"
// CHECK-NOT:              arg:
// CHECK-SAME:            )
  const wchar_t c = L'x';
  wchar_t d = c;
}

namespace pr9608 { // also pr9600
struct incomplete;
incomplete (*x)[3];
}

namespace pr16214 {
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "a"
// CHECK-SAME:             elements: [[A_MEM:![0-9]+]]
// CHECK-SAME:             identifier: "_ZTSN7pr162141aE"
// CHECK: [[A_MEM]] = !{[[A_I:![0-9]*]]}
struct a {
// CHECK: [[A_I]] = !DIDerivedType(tag: DW_TAG_member, name: "i"
  int i;
};

typedef a at;

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "b"
// CHECK-SAME:             DIFlagFwdDecl
struct b {
};

typedef b bt;

void func() {
  at a_inst;
  bt *b_ptr_inst;
  const bt *b_cnst_ptr_inst;
}

}
