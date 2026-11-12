// RUN: %clang_cc1 -std=c++1z %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

template<typename T> struct A {
  A(T = 0);
  A(void*);
};

template<typename T> A(T*) -> A<long>;
A() -> A<int>;

// CHECK-LABEL: @_Z1fPi(
void f(int *p) {
  // CHECK: @_ZN1AIiEC
  A a{};

  // CHECK: @_ZN1AIlEC
  A b = p;

  // CHECK: @_ZN1AIxEC
  A c = 123LL;
}

namespace N {
  template<typename T> struct B { B(T); };
}
using N::B;

struct X {
  template<typename T> struct C { C(T); };
};

// CHECK: @_Z1gIiEDaT_DTcv1AfL0p_ES1_IS0_E
template<typename T> auto g(T x, decltype(A(x)), A<T>) {}
// CHECK: @_Z1hIiEDaT_DTcvN1N1BEfL0p_ENS2_IS0_EE
template<typename T> auto h(T x, decltype(B(x)), B<T>) {}
// CHECK: @_Z1iI1XiEDaT0_DTcvNT_1CEfL0p_ENS2_1CIS1_EE(
template<typename U, typename T> auto i(T x, decltype(typename U::C(x)), typename U::template C<T>) {}
void test() {
  g(1, 2, A(3));
  h(1, 2, B(3));
  i<X>(1, 2, X::C(3));
}
