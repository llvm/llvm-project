// RUN: %clang_cc1 -std=c++2a -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s

#include "Inputs/std-compare.h"

// CHECK: @_ZTV1A =
struct A;
struct X {
  // CHECK-SAME: @_ZN1X1xEv
  virtual void x();
  friend auto operator<=>(X, X) = default;
};
struct Y {
  virtual ~Y();
  virtual A &operator=(const A &);
  friend auto operator<=>(Y, Y) = default;
};
struct A : X, Y {
  // CHECK-SAME: @_ZN1A1fEv
  virtual void f();
  // CHECK-SAME: @_ZNKR1AssERKS_
  virtual std::strong_ordering operator<=>(const A &) const & = default;
  // CHECK-SAME: @_ZN1A1gEv
  virtual void g();
  // CHECK-SAME: @_ZN1A1hEv
  virtual void h();

  // CHECK-SAME: @_ZN1AaSERKS_
  // implicit virtual A &operator=(const A&) = default;

  // CHECK-SAME: @_ZN1AD1Ev
  // CHECK-SAME: @_ZN1AD0Ev
  // implicit virtual ~A();

  // CHECK-SAME: @_ZNKR1AeqERKS_
  // implicit virtual A &operator==(const A&) const & = default;
};

// For Y:
// CHECK-SAME: @_ZTI1A

// CHECK-SAME: @_ZThn{{[0-9]*}}_N1AD1Ev
// CHECK-SAME: @_ZThn{{[0-9]*}}_N1AD0Ev
// virtual ~Y();

// CHECK-SAME: @_ZThn{{[0-9]*}}_N1AaSERKS_
// virtual A &operator=(const A &);

void A::f() {}
