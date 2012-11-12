// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// PR5695

struct A { A(const A&); ~A(); };
A& a();
void b() {
  A x = a();
}

// CHECK: call {{.*}} @_ZN1AC1ERKS_
// CHECK: call {{.*}} @_ZN1AD1Ev
