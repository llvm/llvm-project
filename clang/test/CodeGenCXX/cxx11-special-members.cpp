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

// rdar://18309639 {
template<int> struct C { C() = default; };
struct D {
  C<0> c;
  D() { }
};
template struct C<0>; // was asserting
void f3() {
  C<0> a;
  D b;
}
// Trivial default ctor, might or might not be defined, but we must not expect
// someone else ot define it.
// CHECK-NOT: declare {{.*}} @_ZN1CILi0EEC1Ev
// CHECK: define {{.*}} @_ZN1DC1Ev

// CHECK: define {{.*}} @_ZN1BC2EOS_(
// CHECK: call {{.*}} @_ZN1AC1ERKS_(
