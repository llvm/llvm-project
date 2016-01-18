// Ensure that implicit methods aren't instrumented.

// RUN: %clang_cc1 -x c++ %s -triple %itanium_abi_triple -main-file-name cxx-implicit.cpp -o - -emit-llvm -fprofile-instr-generate | FileCheck %s

// An implicit constructor is generated for Base. We should not emit counters
// for it.
// CHECK-NOT: @__llvm_profile_counters__ZN4BaseC2Ev =

struct Base {
  virtual void foo();
};

struct Derived : public Base {
  Derived();
};

Derived::Derived() {}
