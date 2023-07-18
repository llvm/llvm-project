// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s

// <rdar://problem/8684363>: clang++ not respecting __attribute__((used)) on destructors
struct X0 {
  // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X0C1Ev
  __attribute__((used)) X0() {}
  // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X0D1Ev
  __attribute__((used)) ~X0() {}
};

// PR19743: not emitting __attribute__((used)) inline methods in nested classes.
struct X1 {
  struct Nested {
    // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X16Nested1fEv
    void __attribute__((used)) f() {}
  };
};

struct X2 {
  // We must delay emission of bar() until foo() has had its body parsed,
  // otherwise foo() would not be emitted.
  void __attribute__((used)) bar() { foo(); }
  void foo() { }

  // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X23barEv
  // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X23fooEv
};
