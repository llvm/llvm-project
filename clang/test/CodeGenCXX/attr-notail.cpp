// RUN: %clang_cc1 -triple=x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

class Class1 {
public:
  [[clang::not_tail_called]] int m1();
  int m2();
};

int foo1(int a, Class1 *c1) {
  if (a)
    return c1->m1();
  return c1->m2();
}

// CHECK-LABEL: define{{.*}} i32 @_Z4foo1iP6Class1(
// CHECK: %{{[a-z0-9]+}} = notail call i32 @_ZN6Class12m1Ev(%class.Class1*
// CHECK: %{{[a-z0-9]+}} = call i32 @_ZN6Class12m2Ev(%class.Class1*
