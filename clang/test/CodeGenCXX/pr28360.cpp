// RUN: %clang_cc1 %s -emit-llvm -o - -triple i686-pc-win32 | FileCheck %s
struct A {
  void Foo();
  void Foo(int);
};

using MpTy = void (A::*)();

void Bar(const MpTy &);

void Baz() { Bar(&A::Foo); }

// CHECK-LABEL: define void @"\01?Baz@@YAXXZ"(
// CHECK:  %[[ref_tmp:.*]] = alloca i8*, align 4
// CHECK: store i8* bitcast (void (%struct.A*)* @"\01?Foo@A@@QAEXXZ" to i8*), i8** %[[ref_tmp]], align 4
// CHECK: call void @"\01?Bar@@YAXABQ8A@@AEXXZ@Z"(i8** dereferenceable(4) %[[ref_tmp]])
