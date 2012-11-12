// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// <rdar://problem/8684363>: clang++ not respecting __attribute__((used)) on destructors
struct X0 {
  // CHECK: define linkonce_odr {{.*}} @_ZN2X0C1Ev
  __attribute__((used)) X0() {}
  // CHECK: define linkonce_odr {{.*}} @_ZN2X0D1Ev
  __attribute__((used)) ~X0() {}
};
