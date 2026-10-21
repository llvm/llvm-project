// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-linux-gnu -std=c++2a | FileCheck %s

namespace spaceship {
  struct X {};
  struct Y {};
  int operator<=>(X, Y);

  // CHECK-LABEL: define {{.*}} @_ZN9spaceship1fIiEEvDTcmltcvNS_1YE_EcvNS_1XE_EcvT__EE
  template<typename T> void f(decltype(Y() < X(), T()) x) {}
  template void f<int>(int);
}

namespace paren_init {
  struct A { int a; };
  struct B { int a, b; };
  struct D { int a, b = 7; };
  template <int> struct X {};

  template <int N> void f(X<N + A(1).a>) {}
  template <int N> void f(X<N + A(2).a>) {}

  // CHECK-LABEL: define {{.*}} @_ZN10paren_init1fILi0EEEvNS_1XIXplT_dtcvNS_1AELi1E1aEEE
  template void f<0>(X<1>);
  // CHECK-LABEL: define {{.*}} @_ZN10paren_init1fILi0EEEvNS_1XIXplT_dtcvNS_1AELi2E1aEEE
  template void f<0>(X<2>);

  template <int N> void c_style(X<N + ((A)1).a>) {}
  template <int N> void static_style(X<N + static_cast<A>(1).a>) {}

  // CHECK-LABEL: define {{.*}} @_ZN10paren_init7c_styleILi0EEEvNS_1XIXplT_dtcvNS_1AELi1E1aEEE
  template void c_style<0>(X<1>);
  // CHECK-LABEL: define {{.*}} @_ZN10paren_init12static_styleILi0EEEvNS_1XIXplT_dtscNS_1AELi1E1aEEE
  template void static_style<0>(X<1>);

  // Check zero and multiple arguments, and a default member initializer.
  template <int N> void zero(X<N + A().a>) {}
  template <int N> void two(X<N + B(1, 2).b>) {}
  template <int N> void defaulted(X<N + D(1).b>) {}

  // CHECK-LABEL: define {{.*}} @_ZN10paren_init4zeroILi0EEEvNS_1XIXplT_dtcvNS_1AE_E1aEEE
  template void zero<0>(X<0>);
  // CHECK-LABEL: define {{.*}} @_ZN10paren_init3twoILi0EEEvNS_1XIXplT_dtcvNS_1BE_Li1ELi2EE1bEEE
  template void two<0>(X<2>);
  // CHECK-LABEL: define {{.*}} @_ZN10paren_init9defaultedILi0EEEvNS_1XIXplT_dtcvNS_1DELi1E1bEEE
  template void defaulted<0>(X<7>);

  template <int N> void new_zero(decltype((N, new A()))) {}
  template <int N> void new_one(decltype((N, new A(1)))) {}
  template <int N> void new_two(decltype((N, new B(1, 2)))) {}

  // CHECK-LABEL: define {{.*}} @_ZN10paren_init8new_zeroILi0EEEvDTcmT_nw_NS_1AEpiEE
  template void new_zero<0>(A *);
  // CHECK-LABEL: define {{.*}} @_ZN10paren_init7new_oneILi0EEEvDTcmT_nw_NS_1AEpiLi1EEE
  template void new_one<0>(A *);
  // CHECK-LABEL: define {{.*}} @_ZN10paren_init7new_twoILi0EEEvDTcmT_nw_NS_1BEpiLi1ELi2EEE
  template void new_two<0>(B *);
}
