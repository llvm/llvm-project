// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s %std_cxx11-14 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s %std_cxx17-   | FileCheck %s
// expected-no-diagnostics
// PR47655
template <typename T> struct S {
  S(int, T *) {}
};

template <typename T>
int also_before(T s) {
  return 0;
}

#pragma omp begin declare variant match(implementation = {extension(allow_templates)})
template <typename T>
int also_before(S<T> s) {
  // Ensure there is no error because this is never instantiated.
  double t;
  S<T> q(1, &t);
  return 1;
}
template <typename T>
int special(S<T> s) {
  T t;
  S<T> q(0, &t);
  return 0;
}
template <typename T>
int also_after(S<T> s) {
  // Ensure there is no error because this is never instantiated.
  double t;
  S<T> q(2.0, &t);
  return 2;
}
#pragma omp end declare variant

template <typename T>
int also_after(T s) {
  return 0;
}

int test() {
  // Should return 0.
  return also_before(0) + also_after(0) + also_before(0.) + also_after(0.) + special(S<int>(0, 0));
}

// CHECK: call {{.*}} @_Z11also_beforeIiEiT_
// CHECK: call {{.*}} @_Z10also_afterIiEiT_
// CHECK: call {{.*}} @_Z11also_beforeIdEiT_
// CHECK: call {{.*}} @_Z10also_afterIdEiT_
// CHECK: call {{.*}} @"_Z42special$ompvariant$S4$s12$Pallow_templatesIiEi1SIT_E"
