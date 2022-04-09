// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple i686-pc-win32 | FileCheck %s
struct A {
  void Foo();
  void Foo(int);
};

using MpTy = void (A::*)();

void Bar(const MpTy &);

void Baz() { Bar(&A::Foo); }

// CHECK-LABEL: define dso_local void @"?Baz@@YAXXZ"(
// CHECK:  %[[ref_tmp:.*]] = alloca i8*, align 4
// CHECK: store i8* bitcast (void (%struct.A*)* @"?Foo@A@@QAEXXZ" to i8*), i8** %[[ref_tmp]], align 4
// CHECK: call void @"?Bar@@YAXABQ8A@@AEXXZ@Z"(i8** noundef nonnull align 4 dereferenceable(4) %[[ref_tmp]])
