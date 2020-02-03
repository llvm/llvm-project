// RUN: %clang_cc1 -triple=i686 -emit-llvm -o - %s | FileCheck %s


struct X;
typedef void (X::*memptr)();

struct A {
  virtual memptr f();
};

struct B {
  virtual memptr f();
};

struct C : A, B {
  C();
  memptr f() override __attribute__((noinline)) { return nullptr; };
};

C::C() {}

// Make sure the member pointer is returned from the thunk via the return slot.
// Because of the tail call, the return value cannot be copied into a local
// alloca. (PR39901)

// CHECK-LABEL: define linkonce_odr void @_ZThn4_N1C1fEv({ i32, i32 }* noalias sret %agg.result, %struct.C* %this)
// CHECK: tail call void @_ZN1C1fEv({ i32, i32 }* sret %agg.result
