// RUN: %clang_cc1 %s -emit-llvm -o - -triple i686-pc-win32 | FileCheck %s
struct A {
  void Foo();
  void Foo(int);
};

using MpTy = void (A::*)();

void Bar(const MpTy &);

void Baz() { Bar(&A::Foo); }

// CHECK-LABEL: define dso_local void @"?Baz@@YAXXZ"(
// CHECK:  %[[ref_tmp:.*]] = alloca ptr, align 4
// CHECK: store ptr @"?Foo@A@@QAEXXZ", ptr %[[ref_tmp]], align 4
// CHECK: call void @"?Bar@@YAXABQ8A@@AEXXZ@Z"(ptr noundef nonnull align 4 dereferenceable(4) %[[ref_tmp]])
