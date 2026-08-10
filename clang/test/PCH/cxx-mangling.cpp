// Test without PCH.
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -include %s %s -emit-llvm -o - | FileCheck %s
//
// Test with PCH.
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -x c++-header %s -emit-pch -o %t
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -include-pch %t %s -emit-llvm -o - | FileCheck %s

#ifndef HEADER
#define HEADER

struct A {
  struct { int a; } a;
  struct { int b; } b;
};

#else

template<typename T> void f(T) {}

// CHECK-LABEL: define {{.*}}void @_Z1g1A(
void g(A a) {
  // CHECK: call {{.*}}void @_Z1fIN1AUt0_EEvT_(
  f(a.b);
  // CHECK: call {{.*}}void @_Z1fIN1AUt_EEvT_(
  f(a.a);
}

#endif
