// RUN: %clang_cc1 -std=c++11 -emit-llvm -o - %s | FileCheck %s

template <typename T>
struct X {
  X();
};

// CHECK: define {{.*}} @_ZN1XIbEC1Ev
// CHECK: define {{.*}} @_ZN1XIbEC2Ev
template <> X<bool>::X() = default;

// CHECK: define weak_odr {{.*}} @_ZN1XIiEC1Ev
// CHECK: define weak_odr {{.*}} @_ZN1XIiEC2Ev
template <typename T> X<T>::X() = default;
template X<int>::X();

// CHECK: define linkonce_odr {{.*}} @_ZN1XIcEC1Ev
// CHECK: define linkonce_odr {{.*}} @_ZN1XIcEC2Ev
X<char> x;
