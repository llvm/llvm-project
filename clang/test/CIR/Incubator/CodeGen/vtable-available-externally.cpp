// RUN: %clang_cc1 %s -I%S -I%S/../Inputs -triple x86_64-unknown-linux-gnu -std=c++98 -O0 -disable-llvm-passes -emit-cir -o %t
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK %s < %t
// RUN: %clang_cc1 %s -I%S -I%S/../Inputs -triple x86_64-unknown-linux-gnu -std=c++98 -O2 -disable-llvm-passes -emit-cir -o %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-FORCE-EMIT %s < %t.opt

#include <typeinfo>

// CHECK: cir.global{{.*}} external @_ZTV1A
// CHECK-FORCE-EMIT-DAG: cir.global{{.*}} available_externally @_ZTV1A
struct A {
  A();
  virtual void f();
  virtual ~A() { }
};

A::A() { }

void f(A* a) {
  a->f();
};
void g() {
  A a;
  f(&a);
}
