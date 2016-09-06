// RUN: %clang_cc1 %s -fexceptions -fcxx-exceptions -std=c++11 -O1 -triple x86_64 -emit-llvm -o - | FileCheck %s

// lifetime.end should be invoked even if the destructor doesn't run due to an
// exception thrown from previous ctor call.

struct A { A(); ~A(); };
A Baz(const A&);

void Test1() {
  // CHECK-LABEL: @_Z5Test1v(
  // CHECK: getelementptr
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 1, i8* [[TMP:[^ ]+]])
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 1, i8* [[TMP1:[^ ]+]])

  // Normal exit
  // CHECK: call void @llvm.lifetime.end(i64 1, i8* [[TMP1]])
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 1, i8* [[TMP]])

  // Exception exit
  // CHECK: call void @llvm.lifetime.end(i64 1, i8* [[TMP1]])
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 1, i8* [[TMP]])
  Baz(Baz(A()));
}
