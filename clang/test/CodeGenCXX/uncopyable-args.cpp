// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-msvc -emit-llvm -o - %s | FileCheck %s -check-prefix=WIN64

namespace trivial {
// Trivial structs should be passed directly.
struct A {
  void *p;
};
void foo(A);
void bar() {
  foo({});
}
// CHECK-LABEL: define void @_ZN7trivial3barEv()
// CHECK: alloca %"struct.trivial::A"
// CHECK: load i8*, i8**
// CHECK: call void @_ZN7trivial3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN7trivial3fooENS_1AE(i8*)

// WIN64-LABEL: declare void @"\01?foo@trivial@@YAXUA@1@@Z"(i64)
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
// CHECK-LABEL: define void @_ZN12default_ctor3barEv()
// CHECK: alloca %"struct.default_ctor::A"
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK: load i8*, i8**
// CHECK: call void @_ZN12default_ctor3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN12default_ctor3fooENS_1AE(i8*)

// WIN64-LABEL: declare void @"\01?foo@default_ctor@@YAXUA@1@@Z"(i64)
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
// FIXME: The copy ctor is implicitly deleted.
// CHECK-DISABLED-LABEL: define void @_ZN9move_ctor3barEv()
// CHECK-DISABLED: call void @_Z{{.*}}C1Ev(
// CHECK-DISABLED-NOT: call
// CHECK-DISABLED: call void @_ZN9move_ctor3fooENS_1AE(%"struct.move_ctor::A"* %{{.*}})
// CHECK-DISABLED-LABEL: declare void @_ZN9move_ctor3fooENS_1AE(%"struct.move_ctor::A"*)

// WIN64-LABEL: declare void @"\01?foo@move_ctor@@YAXUA@1@@Z"(%"struct.move_ctor::A"*)
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
// FIXME: The copy ctor is deleted.
// CHECK-DISABLED-LABEL: define void @_ZN11all_deleted3barEv()
// CHECK-DISABLED: call void @_Z{{.*}}C1Ev(
// CHECK-DISABLED-NOT: call
// CHECK-DISABLED: call void @_ZN11all_deleted3fooENS_1AE(%"struct.all_deleted::A"* %{{.*}})
// CHECK-DISABLED-LABEL: declare void @_ZN11all_deleted3fooENS_1AE(%"struct.all_deleted::A"*)

// WIN64-LABEL: declare void @"\01?foo@all_deleted@@YAXUA@1@@Z"(%"struct.all_deleted::A"*)
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
// FIXME: The copy and move ctors are implicitly deleted.
// CHECK-DISABLED-LABEL: define void @_ZN18implicitly_deleted3barEv()
// CHECK-DISABLED: call void @_Z{{.*}}C1Ev(
// CHECK-DISABLED-NOT: call
// CHECK-DISABLED: call void @_ZN18implicitly_deleted3fooENS_1AE(%"struct.implicitly_deleted::A"* %{{.*}})
// CHECK-DISABLED-LABEL: declare void @_ZN18implicitly_deleted3fooENS_1AE(%"struct.implicitly_deleted::A"*)

// WIN64-LABEL: declare void @"\01?foo@implicitly_deleted@@YAXUA@1@@Z"(%"struct.implicitly_deleted::A"*)
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
// FIXME: The copy constructor is implicitly deleted.
// CHECK-DISABLED-LABEL: define void @_ZN11one_deleted3barEv()
// CHECK-DISABLED: call void @_Z{{.*}}C1Ev(
// CHECK-DISABLED-NOT: call
// CHECK-DISABLED: call void @_ZN11one_deleted3fooENS_1AE(%"struct.one_deleted::A"* %{{.*}})
// CHECK-DISABLED-LABEL: declare void @_ZN11one_deleted3fooENS_1AE(%"struct.one_deleted::A"*)

// WIN64-LABEL: declare void @"\01?foo@one_deleted@@YAXUA@1@@Z"(%"struct.one_deleted::A"*)
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
// CHECK-LABEL: define void @_ZN14copy_defaulted3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK: load i8*, i8**
// CHECK: call void @_ZN14copy_defaulted3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN14copy_defaulted3fooENS_1AE(i8*)

// WIN64-LABEL: declare void @"\01?foo@copy_defaulted@@YAXUA@1@@Z"(i64)
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
// CHECK-LABEL: define void @_ZN14move_defaulted3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK: load i8*, i8**
// CHECK: call void @_ZN14move_defaulted3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN14move_defaulted3fooENS_1AE(i8*)

// WIN64-LABEL: declare void @"\01?foo@move_defaulted@@YAXUA@1@@Z"(%"struct.move_defaulted::A"*)
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
// CHECK-LABEL: define void @_ZN17trivial_defaulted3barEv()
// CHECK: call void @_Z{{.*}}C1Ev(
// CHECK: load i8*, i8**
// CHECK: call void @_ZN17trivial_defaulted3fooENS_1AE(i8* %{{.*}})
// CHECK-LABEL: declare void @_ZN17trivial_defaulted3fooENS_1AE(i8*)

// WIN64-LABEL: declare void @"\01?foo@trivial_defaulted@@YAXUA@1@@Z"(i64)
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
// FIXME: This class has a non-trivial copy ctor and a trivial copy ctor.  It's
// not clear whether we should pass by address or in registers.
// CHECK-DISABLED-LABEL: define void @_ZN14two_copy_ctors3barEv()
// CHECK-DISABLED: call void @_Z{{.*}}C1Ev(
// CHECK-DISABLED: call void @_ZN14two_copy_ctors3fooENS_1BE(%"struct.two_copy_ctors::B"* %{{.*}})
// CHECK-DISABLED-LABEL: declare void @_ZN14two_copy_ctors3fooENS_1BE(%"struct.two_copy_ctors::B"*)

// WIN64-LABEL: declare void @"\01?foo@two_copy_ctors@@YAXUB@1@@Z"(%"struct.two_copy_ctors::B"*)
}

namespace definition_only {
struct A {
  A();
  A(A &&o);
  void *p;
};
void *foo(A a) { return a.p; }
// WIN64-LABEL: define i8* @"\01?foo@definition_only@@YAPEAXUA@1@@Z"(%"struct.definition_only::A"*
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
// WIN64-LABEL: define i8* @"\01?foo@deleted_by_member@@YAPEAXUA@1@@Z"(%"struct.deleted_by_member::A"*
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
// WIN64-LABEL: define i8* @"\01?foo@deleted_by_base@@YAPEAXUA@1@@Z"(%"struct.deleted_by_base::A"*
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
// WIN64-LABEL: define i8* @"\01?foo@deleted_by_member_copy@@YAPEAXUA@1@@Z"(%"struct.deleted_by_member_copy::A"*
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
// WIN64-LABEL: define i8* @"\01?foo@deleted_by_base_copy@@YAPEAXUA@1@@Z"(%"struct.deleted_by_base_copy::A"*
}

namespace explicit_delete {
struct A {
  A();
  A(const A &o) = delete;
  void *p;
};
// WIN64-LABEL: define i8* @"\01?foo@explicit_delete@@YAPEAXUA@1@@Z"(%"struct.explicit_delete::A"*
void *foo(A a) { return a.p; }
}
