// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=NEWABI
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown -fclang-abi-compat=4.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=OLDABI
// RUN: %clang_cc1 -std=c++11 -triple x86_64-scei-ps4 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=OLDABI
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-msvc -emit-llvm -o - %s -fms-compatibility -fms-compatibility-version=18 | FileCheck %s -check-prefix=WIN64 -check-prefix=WIN64-18
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-msvc -emit-llvm -o - %s -fms-compatibility -fms-compatibility-version=19 | FileCheck %s -check-prefix=WIN64 -check-prefix=WIN64-19

namespace trivial {
// Trivial structs should be passed directly.
struct A {
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CHECK-LABEL: define{{.*}} void @_ZN7trivial3barEv()
// CHECK: alloca %"struct.trivial::A"
// CHECK: load i8*, i8**
// CHECK: call void @_ZN7trivial3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN7trivial3fooENS_1AE(i8*)

// WIN64-LABEL: declare dso_local void @"?foo@trivial@@YAXUA@1@@Z"(i64)
}

namespace default_ctor {
struct A {
  A();
  void *p;
};
void foo(A);
void bar() {
  // Core issue 1590.  We can pass this type in registers, even though C++
  // normally doesn't permit copies when using braced initialization.
  foo({});
}
// CHECK-LABEL: define{{.*}} void @_ZN12default_ctor3barEv()
// CHECK: alloca %"struct.default_ctor::A"
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK: load i8*, i8**
// CHECK: call void @_ZN12default_ctor3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN12default_ctor3fooENS_1AE(i8*)

// WIN64-LABEL: declare dso_local void @"?foo@default_ctor@@YAXUA@1@@Z"(i64)
}

namespace move_ctor {
// The presence of a move constructor implicitly deletes the trivial copy ctor
// and means that we have to pass this struct by address.
struct A {
  A();
  A(A &&o);
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CHECK-LABEL: define{{.*}} void @_ZN9move_ctor3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK-NOT: call
// NEWABI: call void @_ZN9move_ctor3fooENS_1AE(%"struct.move_ctor::A"* noundef %{{.*}})
// OLDABI: call void @_ZN9move_ctor3fooENS_1AE(i8* %{{.*}})
// NEWABI-LABEL: declare void @_ZN9move_ctor3fooENS_1AE(%"struct.move_ctor::A"* noundef)
// OLDABI-LABEL: declare void @_ZN9move_ctor3fooENS_1AE(i8*)

// WIN64-LABEL: declare dso_local void @"?foo@move_ctor@@YAXUA@1@@Z"(%"struct.move_ctor::A"* noundef)
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
// CHECK-LABEL: define{{.*}} void @_ZN11all_deleted3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK-NOT: call
// NEWABI: call void @_ZN11all_deleted3fooENS_1AE(%"struct.all_deleted::A"* noundef %{{.*}})
// OLDABI: call void @_ZN11all_deleted3fooENS_1AE(i8* %{{.*}})
// NEWABI-LABEL: declare void @_ZN11all_deleted3fooENS_1AE(%"struct.all_deleted::A"* noundef)
// OLDABI-LABEL: declare void @_ZN11all_deleted3fooENS_1AE(i8*)

// WIN64-LABEL: declare dso_local void @"?foo@all_deleted@@YAXUA@1@@Z"(%"struct.all_deleted::A"* noundef)
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
// CHECK-LABEL: define{{.*}} void @_ZN18implicitly_deleted3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK-NOT: call
// NEWABI: call void @_ZN18implicitly_deleted3fooENS_1AE(%"struct.implicitly_deleted::A"* noundef %{{.*}})
// OLDABI: call void @_ZN18implicitly_deleted3fooENS_1AE(i8* %{{.*}})
// NEWABI-LABEL: declare void @_ZN18implicitly_deleted3fooENS_1AE(%"struct.implicitly_deleted::A"* noundef)
// OLDABI-LABEL: declare void @_ZN18implicitly_deleted3fooENS_1AE(i8*)

// In MSVC 2013, the copy ctor is not deleted by a move assignment. In MSVC 2015, it is.
// WIN64-18-LABEL: declare dso_local void @"?foo@implicitly_deleted@@YAXUA@1@@Z"(i64
// WIN64-19-LABEL: declare dso_local void @"?foo@implicitly_deleted@@YAXUA@1@@Z"(%"struct.implicitly_deleted::A"* noundef)
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
// CHECK-LABEL: define{{.*}} void @_ZN11one_deleted3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK-NOT: call
// NEWABI: call void @_ZN11one_deleted3fooENS_1AE(%"struct.one_deleted::A"* noundef %{{.*}})
// OLDABI: call void @_ZN11one_deleted3fooENS_1AE(i8* %{{.*}})
// NEWABI-LABEL: declare void @_ZN11one_deleted3fooENS_1AE(%"struct.one_deleted::A"* noundef)
// OLDABI-LABEL: declare void @_ZN11one_deleted3fooENS_1AE(i8*)

// WIN64-LABEL: declare dso_local void @"?foo@one_deleted@@YAXUA@1@@Z"(%"struct.one_deleted::A"* noundef)
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
// CHECK-LABEL: define{{.*}} void @_ZN14copy_defaulted3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK: load i8*, i8**
// CHECK: call void @_ZN14copy_defaulted3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN14copy_defaulted3fooENS_1AE(i8*)

// WIN64-LABEL: declare dso_local void @"?foo@copy_defaulted@@YAXUA@1@@Z"(i64)
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
// CHECK-LABEL: define{{.*}} void @_ZN14move_defaulted3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK: load i8*, i8**
// CHECK: call void @_ZN14move_defaulted3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN14move_defaulted3fooENS_1AE(i8*)

// WIN64-LABEL: declare dso_local void @"?foo@move_defaulted@@YAXUA@1@@Z"(%"struct.move_defaulted::A"* noundef)
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
// CHECK-LABEL: define{{.*}} void @_ZN17trivial_defaulted3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK: load i8*, i8**
// CHECK: call void @_ZN17trivial_defaulted3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN17trivial_defaulted3fooENS_1AE(i8*)

// WIN64-LABEL: declare dso_local void @"?foo@trivial_defaulted@@YAXUA@1@@Z"(i64)
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
// CHECK-LABEL: define{{.*}} void @_ZN14two_copy_ctors3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// NEWABI: call void @_ZN14two_copy_ctors3fooENS_1BE(%"struct.two_copy_ctors::B"* noundef %{{.*}})
// OLDABI: call void @_ZN14two_copy_ctors3fooENS_1BE(%"struct.two_copy_ctors::B"* noundef byval
// NEWABI-LABEL: declare void @_ZN14two_copy_ctors3fooENS_1BE(%"struct.two_copy_ctors::B"* noundef)
// OLDABI-LABEL: declare void @_ZN14two_copy_ctors3fooENS_1BE(%"struct.two_copy_ctors::B"* noundef byval

// WIN64-LABEL: declare dso_local void @"?foo@two_copy_ctors@@YAXUB@1@@Z"(%"struct.two_copy_ctors::B"* noundef)
}

namespace definition_only {
struct A {
  A();
  A(A &&o);
  void *p;
};
void *foo(A a) { return a.p; }
// NEWABI-LABEL: define{{.*}} i8* @_ZN15definition_only3fooENS_1AE(%"struct.definition_only::A"*
// OLDABI-LABEL: define{{.*}} i8* @_ZN15definition_only3fooENS_1AE(i8*
// WIN64-LABEL: define dso_local noundef i8* @"?foo@definition_only@@YAPEAXUA@1@@Z"(%"struct.definition_only::A"*
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
// NEWABI-LABEL: define{{.*}} i8* @_ZN17deleted_by_member3fooENS_1AE(%"struct.deleted_by_member::A"*
// OLDABI-LABEL: define{{.*}} i8* @_ZN17deleted_by_member3fooENS_1AE(i8*
// WIN64-LABEL: define dso_local noundef i8* @"?foo@deleted_by_member@@YAPEAXUA@1@@Z"(%"struct.deleted_by_member::A"*
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
// NEWABI-LABEL: define{{.*}} i8* @_ZN15deleted_by_base3fooENS_1AE(%"struct.deleted_by_base::A"*
// OLDABI-LABEL: define{{.*}} i8* @_ZN15deleted_by_base3fooENS_1AE(i8*
// WIN64-LABEL: define dso_local noundef i8* @"?foo@deleted_by_base@@YAPEAXUA@1@@Z"(%"struct.deleted_by_base::A"*
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
// NEWABI-LABEL: define{{.*}} i8* @_ZN22deleted_by_member_copy3fooENS_1AE(%"struct.deleted_by_member_copy::A"*
// OLDABI-LABEL: define{{.*}} i8* @_ZN22deleted_by_member_copy3fooENS_1AE(i8*
// WIN64-LABEL: define dso_local noundef i8* @"?foo@deleted_by_member_copy@@YAPEAXUA@1@@Z"(%"struct.deleted_by_member_copy::A"*
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
// NEWABI-LABEL: define{{.*}} i8* @_ZN20deleted_by_base_copy3fooENS_1AE(%"struct.deleted_by_base_copy::A"*
// OLDABI-LABEL: define{{.*}} i8* @_ZN20deleted_by_base_copy3fooENS_1AE(i8*
// WIN64-LABEL: define dso_local noundef i8* @"?foo@deleted_by_base_copy@@YAPEAXUA@1@@Z"(%"struct.deleted_by_base_copy::A"*
}

namespace explicit_delete {
struct A {
  A();
  A(const A &o) = delete;
  void *p;
};
// NEWABI-LABEL: define{{.*}} i8* @_ZN15explicit_delete3fooENS_1AE(%"struct.explicit_delete::A"*
// OLDABI-LABEL: define{{.*}} i8* @_ZN15explicit_delete3fooENS_1AE(i8*
// WIN64-LABEL: define dso_local noundef i8* @"?foo@explicit_delete@@YAPEAXUA@1@@Z"(%"struct.explicit_delete::A"*
void *foo(A a) { return a.p; }
}

namespace implicitly_deleted_copy_ctor {
struct A {
  // No move ctor due to copy assignment.
  A &operator=(const A&);
  // Deleted copy ctor due to rvalue ref member.
  int &&ref;
};
// NEWABI-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1AE(%"struct.implicitly_deleted_copy_ctor::A"*
// OLDABI-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1AE(i32*
// WIN64-LABEL: define {{.*}} @"?foo@implicitly_deleted_copy_ctor@@YAAEAHUA@1@@Z"(%"struct.implicitly_deleted_copy_ctor::A"*
int &foo(A a) { return a.ref; }

struct B {
  // Passed direct: has non-deleted trivial copy ctor.
  B &operator=(const B&);
  int &ref;
};
int &foo(B b) { return b.ref; }
// CHECK-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1BE(i32*
// WIN64-LABEL: define {{.*}} @"?foo@implicitly_deleted_copy_ctor@@YAAEAHUB@1@@Z"(i64

struct X { X(const X&); };
struct Y { Y(const Y&) = default; };

union C {
  C &operator=(const C&);
  // Passed indirect: copy ctor deleted due to variant member with nontrivial copy ctor.
  X x;
  int n;
};
int foo(C c) { return c.n; }
// CHECK-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1CE(%"union.implicitly_deleted_copy_ctor::C"*
// WIN64-LABEL: define {{.*}} @"?foo@implicitly_deleted_copy_ctor@@YAHTC@1@@Z"(%"union.implicitly_deleted_copy_ctor::C"*

struct D {
  D &operator=(const D&);
  // Passed indirect: copy ctor deleted due to variant member with nontrivial copy ctor.
  union {
    X x;
    int n;
  };
};
int foo(D d) { return d.n; }
// CHECK-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1DE(%"struct.implicitly_deleted_copy_ctor::D"*
// WIN64-LABEL: define {{.*}} @"?foo@implicitly_deleted_copy_ctor@@YAHUD@1@@Z"(%"struct.implicitly_deleted_copy_ctor::D"*

union E {
  // Passed direct: has non-deleted trivial copy ctor.
  E &operator=(const E&);
  Y y;
  int n;
};
int foo(E e) { return e.n; }
// CHECK-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1EE(i32
// WIN64-LABEL: define {{.*}} @"?foo@implicitly_deleted_copy_ctor@@YAHTE@1@@Z"(i32

struct F {
  // Passed direct: has non-deleted trivial copy ctor.
  F &operator=(const F&);
  union {
    Y y;
    int n;
  };
};
int foo(F f) { return f.n; }
// CHECK-LABEL: define {{.*}} @_ZN28implicitly_deleted_copy_ctor3fooENS_1FE(i32
// WIN64-LABEL: define {{.*}} @"?foo@implicitly_deleted_copy_ctor@@YAHUF@1@@Z"(i32
}
