// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - -triple=i686-linux-gnu | FileCheck %s

struct A {
  A(const A&);
  A &operator=(const A&);
};

struct B {
  A a;
  B(B&&) = default;
  B &operator=(B&&) = default;
};

// CHECK: define {{.*}} @_Z2f1
void f1(B &x) {
  // CHECK-NOT: memcpy
  // CHECK: call {{.*}} @_ZN1BC1EOS_(
  B b(static_cast<B&&>(x));
}

// CHECK: define {{.*}} @_Z2f2
void f2(B &x, B &y) {
  // CHECK-NOT: memcpy
  // CHECK: call {{.*}} @_ZN1BaSEOS_(
  x = static_cast<B&&>(y);
}

// CHECK: define {{.*}} @_ZN1BaSEOS_(
// CHECK: call {{.*}} @_ZN1AaSERKS_(

// CHECK: define {{.*}} @_ZN1BC2EOS_(
// CHECK: call {{.*}} @_ZN1AC1ERKS_(
