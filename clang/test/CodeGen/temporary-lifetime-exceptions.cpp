// RUN: %clang_cc1 %s -fexceptions -fcxx-exceptions -std=c++11 -O1 -triple x86_64 -emit-llvm -o - | FileCheck %s

// lifetime.end should be invoked even if the destructor doesn't run due to an
// exception thrown from previous ctor call.

struct A { A(); ~A(); };
A Baz(const A&);

void Test1() {
  // CHECK-LABEL: @_Z5Test1v(
  // CHECK: call void @llvm.lifetime.start.p0(i64 1, ptr nonnull [[TMP:[^ ]+]])
  // CHECK: call void @llvm.lifetime.start.p0(i64 1, ptr nonnull [[TMP1:[^ ]+]])

  // Normal exit
  // CHECK: call void @llvm.lifetime.end.p0(i64 1, ptr nonnull [[TMP1]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 1, ptr nonnull [[TMP]])

  // Exception exit
  // CHECK: call void @llvm.lifetime.end.p0(i64 1, ptr nonnull [[TMP1]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 1, ptr nonnull [[TMP]])
  Baz(Baz(A()));
}
