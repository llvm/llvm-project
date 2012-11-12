// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

struct A {
  virtual ~A();
};

void f(A *a) {
  // CHECK: define {{.*}} @_Z1fP1A
  // CHECK: load
  // CHECK: load
  // CHECK: [[CALLEE:%[a-zA-Z0-9.]*]] = load
  // CHECK: call {{.*}} [[CALLEE]](
  a->~A();
}
