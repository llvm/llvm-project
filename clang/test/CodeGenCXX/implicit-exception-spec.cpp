// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -std=c++11 -o - -fcxx-exceptions -fexceptions | FileCheck -check-prefix=CHECK %s

struct A {
  A();
  A(const A&);
  A(A&&);
};
struct B : virtual A {
  virtual void f() = 0;
};
struct C : B {
  void f();
};

// CHECK-DAG: define {{.*}} @_ZN1BC2Ev({{.*}} #[[NOUNWIND:[0-9]*]]
C c1;
// CHECK-DAG: define {{.*}} @_ZN1BC2ERKS_({{.*}} #[[NOUNWIND]]
C c2(c1);
// CHECK-DAG: define {{.*}} @_ZN1BC2EOS_({{.*}} #[[NOUNWIND]]
C c3(static_cast<C&&>(c1));

// CHECK-DAG: #[[NOUNWIND]] = {{{.*}} nounwind
