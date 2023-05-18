// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-linux-gnu -std=c++2a | FileCheck %s

namespace spaceship {
  struct X {};
  struct Y {};
  int operator<=>(X, Y);

  // CHECK-LABEL: define {{.*}} @_ZN9spaceship1fIiEEvDTcmltcvNS_1YE_EcvNS_1XE_EcvT__EE
  template<typename T> void f(decltype(Y() < X(), T()) x) {}
  template void f<int>(int);
}
