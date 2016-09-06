// Ensure that implicit methods aren't instrumented.

// RUN: %clang_cc1 -x c++ -std=c++11 %s -triple %itanium_abi_triple -main-file-name cxx-implicit.cpp -o - -emit-llvm -fprofile-instrument=clang | FileCheck %s

// Implicit constructors are generated for Base. We should not emit counters
// for them.
// CHECK-DAG: define {{.*}}_ZN4BaseC2Ev
// CHECK-DAG: define {{.*}}_ZN4BaseC2ERKS_
// CHECK-DAG: define {{.*}}_ZN4BaseC2EOS_
// CHECK-DAG: __profc__ZN7DerivedC2Ev,
// CHECK-DAG: __profc__ZN7DerivedC2ERKS_
// CHECK-DAG: __profc__ZN7DerivedC2EOS_
// CHECK-NOT: @__profc__ZN4BaseC2Ev =
// CHECK-NOT: @__profc__ZN4BaseC2ERKS_
// CHECK-NOT: @__profc__ZN4BaseC2EOS_
//
// Implicit assignment operators are generated for Base. We should not emit counters
// for them.
// CHECK-NOT: @__profc__ZN4BaseaSEOS_
// CHECK-NOT: @__profc__ZN4BaseaSERKS_

struct BaseBase {
 BaseBase();
 BaseBase(const BaseBase &);
 BaseBase &operator=(const BaseBase &);
 BaseBase &operator=(BaseBase &&);
};

struct Base : public BaseBase {
  virtual void foo();
};

struct Derived : public Base {
  Derived();
  Derived(const Derived &);
  Derived(Derived &&);
  Derived &operator=(const Derived &);
  Derived &operator=(Derived &&);
};

Derived::Derived() {}
Derived::Derived(const Derived &d) : Base(d) {}
Derived::Derived(Derived &&d) : Base(static_cast<Base&&>(d)) {}
Derived& Derived::operator=(const Derived &d) {
  Base::operator=(d);
  return *this;
}
Derived& Derived::operator=(Derived &&d) {
  Base::operator=(static_cast<Base &&>(d));
  return *this;
}
