// RUN: %clang_cc1 -std=c++14 %s -emit-llvm -o - -triple nvptx \
// RUN:   -fcuda-is-device | FileCheck --check-prefixes=CXX14 %s
// RUN: %clang_cc1 -std=c++17 %s -emit-llvm -o - -triple nvptx \
// RUN:   -fcuda-is-device | FileCheck --check-prefixes=CXX17 %s

#include "Inputs/cuda.h"

// COM: @_ZL1a = internal {{.*}}constant i32 7
constexpr int a = 7;
__constant__ const int &use_a = a;

namespace B {
 // COM: @_ZN1BL1bE = internal {{.*}}constant i32 9
  constexpr int b = 9;
}
__constant__ const int &use_B_b = B::b;

struct Q {
  // CXX14: @_ZN1Q2k2E = {{.*}}externally_initialized constant i32 6
  // CXX17: @_ZN1Q2k2E = internal {{.*}}constant i32 6
  // CXX14: @_ZN1Q2k1E = available_externally {{.*}}constant i32 5
  // CXX17: @_ZN1Q2k1E = linkonce_odr {{.*}}constant i32 5
  static constexpr int k1 = 5;
  static constexpr int k2 = 6;
};
constexpr int Q::k2;

__constant__ const int &use_Q_k1 = Q::k1;
__constant__ const int &use_Q_k2 = Q::k2;

template<typename T> struct X {
  // CXX14: @_ZN1XIiE1aE = available_externally {{.*}}constant i32 123
  // CXX17: @_ZN1XIiE1aE = linkonce_odr {{.*}}constant i32 123
  static constexpr int a = 123;
};
__constant__ const int &use_X_a = X<int>::a;

template <typename T, T a, T b> struct A {
  // CXX14: @_ZN1AIiLi1ELi2EE1xE = available_externally {{.*}}constant i32 2
  // CXX17: @_ZN1AIiLi1ELi2EE1xE = linkonce_odr {{.*}}constant i32 2
  constexpr static T x = a * b;
};
__constant__ const int &y = A<int, 1, 2>::x;
