// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// All test cases from CodeGenCXX/uncopyable-args.cpp (Itanium x86_64 only).
// Tests CIRGen handling of types with deleted/defaulted/implicit copy/move ctors.

namespace trivial {
struct A {
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN7trivial3barEv
// CIR:   cir.call @_ZN7trivial3fooENS_1AE

// LLVM-LABEL: define {{.*}} void @_ZN7trivial3barEv(
// LLVM:   call void @_ZN7trivial3fooENS_1AE(

// OGCG-LABEL: define {{.*}} void @_ZN7trivial3barEv(
// OGCG:   call void @_ZN7trivial3fooENS_1AE(ptr
}

namespace default_ctor {
struct A {
  A();
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN12default_ctor3barEv
// CIR:   cir.call @_ZN12default_ctor1AC1Ev
// CIR:   cir.call @_ZN12default_ctor3fooENS_1AE

// LLVM-LABEL: define {{.*}} void @_ZN12default_ctor3barEv(
// LLVM:   call void @_ZN12default_ctor1AC1Ev(
// LLVM:   call void @_ZN12default_ctor3fooENS_1AE(

// OGCG-LABEL: define {{.*}} void @_ZN12default_ctor3barEv(
// OGCG:   call {{.*}} @_ZN12default_ctor1AC1Ev(
// OGCG:   call void @_ZN12default_ctor3fooENS_1AE(ptr
}

namespace move_ctor {
struct A {
  A();
  A(A &&o);
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN9move_ctor3barEv
// CIR:   cir.call @_ZN9move_ctor1AC1Ev
// CIR:   cir.call @_ZN9move_ctor3fooENS_1AE

// LLVM-LABEL: define {{.*}} void @_ZN9move_ctor3barEv(
// LLVM:   call void @_ZN9move_ctor1AC1Ev(
// LLVM:   call void @_ZN9move_ctor3fooENS_1AE(

// OGCG-LABEL: define {{.*}} void @_ZN9move_ctor3barEv(
// OGCG:   call {{.*}} @_ZN9move_ctor1AC1Ev(
// OGCG:   call void @_ZN9move_ctor3fooENS_1AE(ptr noundef dead_on_return
}

namespace all_deleted {
struct A {
  A();
  A(const A &o) = delete;
  A(A &&o) = delete;
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN11all_deleted3barEv
// CIR:   cir.call @_ZN11all_deleted1AC1Ev
// CIR:   cir.call @_ZN11all_deleted3fooENS_1AE

// LLVM-LABEL: define {{.*}} void @_ZN11all_deleted3barEv(
// LLVM:   call void @_ZN11all_deleted1AC1Ev(
// LLVM:   call void @_ZN11all_deleted3fooENS_1AE(

// OGCG-LABEL: define {{.*}} void @_ZN11all_deleted3barEv(
// OGCG:   call {{.*}} @_ZN11all_deleted1AC1Ev(
// OGCG:   call void @_ZN11all_deleted3fooENS_1AE(ptr noundef dead_on_return
}

namespace implicitly_deleted {
struct A {
  A();
  A &operator=(A &&o);
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN18implicitly_deleted3barEv
// CIR:   cir.call @_ZN18implicitly_deleted1AC1Ev
// CIR:   cir.call @_ZN18implicitly_deleted3fooENS_1AE

// LLVM-LABEL: define {{.*}} void @_ZN18implicitly_deleted3barEv(
// LLVM:   call void @_ZN18implicitly_deleted1AC1Ev(
// LLVM:   call void @_ZN18implicitly_deleted3fooENS_1AE(

// OGCG-LABEL: define {{.*}} void @_ZN18implicitly_deleted3barEv(
// OGCG:   call {{.*}} @_ZN18implicitly_deleted1AC1Ev(
// OGCG:   call void @_ZN18implicitly_deleted3fooENS_1AE(ptr noundef dead_on_return
}

namespace one_deleted {
struct A {
  A();
  A(A &&o) = delete;
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN11one_deleted3barEv
// CIR:   cir.call @_ZN11one_deleted1AC1Ev
// CIR:   cir.call @_ZN11one_deleted3fooENS_1AE

// LLVM-LABEL: define {{.*}} void @_ZN11one_deleted3barEv(
// LLVM:   call void @_ZN11one_deleted1AC1Ev(
// LLVM:   call void @_ZN11one_deleted3fooENS_1AE(

// OGCG-LABEL: define {{.*}} void @_ZN11one_deleted3barEv(
// OGCG:   call {{.*}} @_ZN11one_deleted1AC1Ev(
// OGCG:   call void @_ZN11one_deleted3fooENS_1AE(ptr noundef dead_on_return
}

namespace copy_defaulted {
struct A {
  A();
  A(const A &o) = default;
  A(A &&o) = delete;
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN14copy_defaulted3barEv
// CIR:   cir.call @_ZN14copy_defaulted1AC1Ev
// CIR:   cir.call @_ZN14copy_defaulted3fooENS_1AE

// LLVM-LABEL: define {{.*}} void @_ZN14copy_defaulted3barEv(
// LLVM:   call void @_ZN14copy_defaulted1AC1Ev(
// LLVM:   call void @_ZN14copy_defaulted3fooENS_1AE(

// OGCG-LABEL: define {{.*}} void @_ZN14copy_defaulted3barEv(
// OGCG:   call {{.*}} @_ZN14copy_defaulted1AC1Ev(
// OGCG:   call void @_ZN14copy_defaulted3fooENS_1AE(ptr
}

namespace move_defaulted {
struct A {
  A();
  A(const A &o) = delete;
  A(A &&o) = default;
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN14move_defaulted3barEv
// CIR:   cir.call @_ZN14move_defaulted1AC1Ev
// CIR:   cir.call @_ZN14move_defaulted3fooENS_1AE

// LLVM-LABEL: define {{.*}} void @_ZN14move_defaulted3barEv(
// LLVM:   call void @_ZN14move_defaulted1AC1Ev(
// LLVM:   call void @_ZN14move_defaulted3fooENS_1AE(

// OGCG-LABEL: define {{.*}} void @_ZN14move_defaulted3barEv(
// OGCG:   call {{.*}} @_ZN14move_defaulted1AC1Ev(
// OGCG:   call void @_ZN14move_defaulted3fooENS_1AE(ptr
}

namespace trivial_defaulted {
struct A {
  A();
  A(const A &o) = default;
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN17trivial_defaulted3barEv
// CIR:   cir.call @_ZN17trivial_defaulted1AC1Ev
// CIR:   cir.call @_ZN17trivial_defaulted3fooENS_1AE

// LLVM-LABEL: define {{.*}} void @_ZN17trivial_defaulted3barEv(
// LLVM:   call void @_ZN17trivial_defaulted1AC1Ev(
// LLVM:   call void @_ZN17trivial_defaulted3fooENS_1AE(

// OGCG-LABEL: define {{.*}} void @_ZN17trivial_defaulted3barEv(
// OGCG:   call {{.*}} @_ZN17trivial_defaulted1AC1Ev(
// OGCG:   call void @_ZN17trivial_defaulted3fooENS_1AE(ptr
}

namespace two_copy_ctors {
struct A {
  A();
  A(const A &) = default;
  A(const A &, int = 0);
  void *p;
};
struct B : A {};

void foo(B);
void bar() {
  foo({});
}
// CIR-LABEL: cir.func {{.*}} @_ZN14two_copy_ctors3barEv
// CIR:   cir.call @{{.*}}C1Ev
// CIR:   cir.call @_ZN14two_copy_ctors3fooENS_1BE

// LLVM-LABEL: define {{.*}} void @_ZN14two_copy_ctors3barEv(
// LLVM:   call void @{{.*}}C1Ev(
// LLVM:   call void @_ZN14two_copy_ctors3fooENS_1BE(

// OGCG-LABEL: define {{.*}} void @_ZN14two_copy_ctors3barEv(
// OGCG:   call {{.*}} @{{.*}}C1Ev(
// OGCG:   call void @_ZN14two_copy_ctors3fooENS_1BE(ptr noundef dead_on_return
}

namespace definition_only {
struct A {
  A();
  A(A &&o);
  void *p;
};
void *foo(A a) { return a.p; }

// CIR-LABEL: cir.func {{.*}} @_ZN15definition_only3fooENS_1AE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN15definition_only3fooENS_1AE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} ptr @_ZN15definition_only3fooENS_1AE(ptr
// OGCG:   ret ptr
}

namespace deleted_by_member {
struct B {
  B();
  B(B &&o);
  void *p;
};
struct A {
  A();
  B b;
};
void *foo(A a) { return a.b.p; }

// CIR-LABEL: cir.func {{.*}} @_ZN17deleted_by_member3fooENS_1AE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN17deleted_by_member3fooENS_1AE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} ptr @_ZN17deleted_by_member3fooENS_1AE(ptr
// OGCG:   ret ptr
}

namespace deleted_by_base {
struct B {
  B();
  B(B &&o);
  void *p;
};
struct A : B {
  A();
};
void *foo(A a) { return a.p; }

// CIR-LABEL: cir.func {{.*}} @_ZN15deleted_by_base3fooENS_1AE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN15deleted_by_base3fooENS_1AE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} ptr @_ZN15deleted_by_base3fooENS_1AE(ptr
// OGCG:   ret ptr
}

namespace deleted_by_member_copy {
struct B {
  B();
  B(const B &o) = delete;
  void *p;
};
struct A {
  A();
  B b;
};
void *foo(A a) { return a.b.p; }

// CIR-LABEL: cir.func {{.*}} @_ZN22deleted_by_member_copy3fooENS_1AE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN22deleted_by_member_copy3fooENS_1AE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} ptr @_ZN22deleted_by_member_copy3fooENS_1AE(ptr
// OGCG:   ret ptr
}

namespace deleted_by_base_copy {
struct B {
  B();
  B(const B &o) = delete;
  void *p;
};
struct A : B {
  A();
};
void *foo(A a) { return a.p; }

// CIR-LABEL: cir.func {{.*}} @_ZN20deleted_by_base_copy3fooENS_1AE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN20deleted_by_base_copy3fooENS_1AE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} ptr @_ZN20deleted_by_base_copy3fooENS_1AE(ptr
// OGCG:   ret ptr
}

namespace explicit_delete {
struct A {
  A();
  A(const A &o) = delete;
  void *p;
};
void *foo(A a) { return a.p; }

// CIR-LABEL: cir.func {{.*}} @_ZN15explicit_delete3fooENS_1AE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN15explicit_delete3fooENS_1AE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} ptr @_ZN15explicit_delete3fooENS_1AE(ptr
// OGCG:   ret ptr
}

namespace implicitly_deleted_copy_ctor {
struct A {
  A &operator=(const A&);
  int &&ref;
};
int &foo(A a) { return a.ref; }

// CIR-LABEL: cir.func {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1AE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1AE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1AE(ptr
// OGCG:   ret ptr

struct B {
  B &operator=(const B&);
  int &ref;
};
int &foo(B b) { return b.ref; }

// CIR-LABEL: cir.func {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1BE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1BE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1BE(ptr
// OGCG:   ret ptr

struct X { X(const X&); };
struct Y { Y(const Y&) = default; };

union C {
  C &operator=(const C&);
  X x;
  int n;
};
int foo(C c) { return c.n; }

// CIR-LABEL: cir.func {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1CE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1CE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1CE(ptr
// OGCG:   ret i32

struct D {
  D &operator=(const D&);
  union {
    X x;
    int n;
  };
};
int foo(D d) { return d.n; }

// CIR-LABEL: cir.func {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1DE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1DE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1DE(ptr
// OGCG:   ret i32

union E {
  E &operator=(const E&);
  Y y;
  int n;
};
int foo(E e) { return e.n; }

// CIR-LABEL: cir.func {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1EE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1EE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} i32 @_ZN28implicitly_deleted_copy_ctor3fooENS_1EE(i32
// OGCG:   ret i32

struct F {
  F &operator=(const F&);
  union {
    Y y;
    int n;
  };
};
int foo(F f) { return f.n; }

// CIR-LABEL: cir.func {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1FE
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1FE(
// LLVM:   ret

// OGCG-LABEL: define {{.*}} i32 @_ZN28implicitly_deleted_copy_ctor3fooENS_1FE(i32
// OGCG:   ret i32
}
