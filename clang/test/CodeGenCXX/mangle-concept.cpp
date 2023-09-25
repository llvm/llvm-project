// RUN: %clang_cc1 -verify -frelaxed-template-template-args -std=c++20 -fclang-abi-compat=latest -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s
// RUN: %clang_cc1 -verify -frelaxed-template-template-args -std=c++20 -fclang-abi-compat=latest -emit-llvm -triple %itanium_abi_triple -o - %s -fclang-abi-compat=16 | FileCheck %s --check-prefix=CLANG16
// expected-no-diagnostics

namespace test1 {
template <bool> struct S {};
template <typename> concept C = true;
template <typename T = int> S<C<T>> f0() { return S<C<T>>{}; }
template S<C<int>> f0<>();
// CHECK: @_ZN5test12f0IiEENS_1SIX1CIT_EEEEv(
// CLANG16: @_ZN5test12f0IiEENS_1SIL_ZNS_1CIT_EEEEEv(
}

template <bool> struct S {};
template <typename> concept C = true;
template <typename, typename> concept D = true;

template <typename T = int> S<test1::C<T>> f0a() { return S<C<T>>{}; }
template S<test1::C<int>> f0a<>();
// CHECK: @_Z3f0aIiE1SIXsr5test1E1CIT_EEEv(
// CLANG16: @_Z3f0aIiE1SIL_ZN5test11CIT_EEEEv(

template <typename T = int> S<C<T>> f0() { return S<C<T>>{}; }
template S<C<int>> f0<>();
// CHECK: @_Z2f0IiE1SIX1CIT_EEEv(
// CLANG16: @_Z2f0IiE1SIL_Z1CIT_EEEv(

template<typename T> concept True = true;

namespace test2 {
  // Member-like friends.
  template<typename T> struct A {
    friend void f(...) requires True<T> {}

    template<typename U = void>
    friend void g(...) requires True<T> && True<U> {}

    template<typename U = void>
    friend void h(...) requires True<U> {}

    template<typename U = void> requires True<T> && True<U>
    friend void i(...) {}

    template<typename U = void> requires True<U>
    friend void j(...) {}

    template<True U = void> requires True<T>
    friend void k(...) {}

    template<True U = void>
    friend void l(...) {}
  };

  A<int> ai;

  // CHECK-LABEL: define {{.*}}@{{.*}}test2{{.*}}use
  void use() {
    // CHECK: call {{.*}}@_ZN5test21AIiEF1fEzQ4TrueIT_E(
    // CLANG16: call {{.*}}@_ZN5test21fEz(
    f(ai);
    // CHECK: call {{.*}}@_ZN5test2F1gIvEEvzQaa4TrueIT_E4TrueITL0__E(
    // CLANG16: call {{.*}}@_ZN5test21gIvEEvz(
    g(ai);
    // CHECK: call {{.*}}@_ZN5test21hIvEEvzQ4TrueITL0__E(
    // CLANG16: call {{.*}}@_ZN5test21hIvEEvz(
    h(ai);
    // CHECK: call {{.*}}@_ZN5test2F1iIvQaa4TrueIT_E4TrueITL0__EEEvz(
    // CLANG16: call {{.*}}@_ZN5test21iIvEEvz(
    i(ai);
    // CHECK: call {{.*}}@_ZN5test21jIvQ4TrueITL0__EEEvz(
    // CLANG16: call {{.*}}@_ZN5test21jIvEEvz(
    j(ai);
    // CHECK: call {{.*}}@_ZN5test2F1kITk4TruevQ4TrueIT_EEEvz(
    // CLANG16: call {{.*}}@_ZN5test21kIvEEvz(
    k(ai);
    // CHECK: call {{.*}}@_ZN5test21lITk4TruevEEvz(
    // CLANG16: call {{.*}}@_ZN5test21lIvEEvz(
    l(ai);
  }
}

namespace test3 {
  // Unconstrained auto.
  template<auto> void d() {}
  template void d<0>();
  // CHECK: define {{.*}}@_ZN5test31dITnDaLi0EEEvv(
  // CLANG16: define {{.*}}@_ZN5test31dILi0EEEvv(

  template<decltype(auto)> void e() {}
  template void e<0>();
  // CHECK: define {{.*}}@_ZN5test31eITnDcLi0EEEvv(
  // CLANG16: define {{.*}}@_ZN5test31eILi0EEEvv(

  // Constrained auto.
  template<C auto> void f() {}
  template void f<0>();
  // CHECK: define {{.*}}@_ZN5test31fITnDk1CLi0EEEvv(
  // CLANG16: define {{.*}}@_ZN5test31fILi0EEEvv(

  template<D<int> auto> void g() {}
  template void g<0>();
  // CHECK: define {{.*}}@_ZN5test31gITnDk1DIiELi0EEEvv(
  // CLANG16: define {{.*}}@_ZN5test31gILi0EEEvv(

  template<typename T, D<T> auto> void h() {}
  template void h<int, 0>();
  // CHECK: define {{.*}}@_ZN5test31hIiTnDk1DIT_ELi0EEEvv(
  // CLANG16: define {{.*}}@_ZN5test31hIiLi0EEEvv(

  template<typename T> void i(decltype(new C auto(T()))) {}
  template void i<int>(int*);
  // CHECK: define {{.*}}@_ZN5test31iIiEEvDTnw_Dk1CpicvT__EEE(
  // CLANG16: define {{.*}}@_ZN5test31iIiEEvDTnw_DapicvT__EEE(

  template<typename T> void j(decltype(new C decltype(auto)(T()))) {}
  template void j<int>(int*);
  // CHECK: define {{.*}}@_ZN5test31jIiEEvDTnw_DK1CpicvT__EEE(
  // CLANG16: define {{.*}}@_ZN5test31jIiEEvDTnw_DcpicvT__EEE(
}

namespace test4 {
  // Constrained type parameters.
  template<C> void f() {}
  template void f<int>();
  // CHECK: define {{.*}}@_ZN5test41fITk1CiEEvv(
  // CLANG16: define {{.*}}@_ZN5test41fIiEEvv(

  template<D<int>> void g() {}
  template void g<int>();
  // CHECK: define {{.*}}@_ZN5test41gITk1DIiEiEEvv(
  // CLANG16: define {{.*}}@_ZN5test41gIiEEvv(
}

namespace test5 {
  // Exact-match vs non-exact-match template template parameters.
  template<typename T, T V> struct X {};
  template<typename T, T V> requires C<T> struct Y {};
  template<C T, T V> struct Z {};

  template<template<typename T, T> typename> void f() {}
  // CHECK: define {{.*}}@_ZN5test51fINS_1XEEEvv(
  template void f<X>();
  // CHECK: define {{.*}}@_ZN5test51fITtTyTnTL0__ENS_1YEEEvv(
  template void f<Y>();
  // CHECK: define {{.*}}@_ZN5test51fITtTyTnTL0__ENS_1ZEEEvv(
  template void f<Z>();

  template<template<typename T, T> requires C<T> typename> void g() {}
  // CHECK: define {{.*}}@_ZN5test51gITtTyTnTL0__Q1CIS1_EENS_1XEEEvv(
  template void g<X>();
  // CHECK: define {{.*}}@_ZN5test51gINS_1YEEEvv(
  template void g<Y>();
  // CHECK: define {{.*}}@_ZN5test51gITtTyTnTL0__Q1CIS1_EENS_1ZEEEvv(
  template void g<Z>();

  template<template<C T, T> typename> void h() {}
  // CHECK: define {{.*}}@_ZN5test51hITtTk1CTnTL0__ENS_1XEEEvv(
  template void h<X>();
  // CHECK: define {{.*}}@_ZN5test51hITtTk1CTnTL0__ENS_1YEEEvv(
  template void h<Y>();
  // CHECK: define {{.*}}@_ZN5test51hINS_1ZEEEvv(
  template void h<Z>();

  // Packs must match the first argument.
  template<template<C T, T> typename...> void i() {}
  // CHECK: define {{.*}}@_ZN5test51iITpTtTk1CTnTL0__EJNS_1XENS_1YENS_1ZEEEEvv(
  template void i<X, Y, Z>();
  // CHECK: define {{.*}}@_ZN5test51iITpTtTk1CTnTL0__EJNS_1YENS_1ZENS_1XEEEEvv(
  template void i<Y, Z, X>();
  // CHECK: define {{.*}}@_ZN5test51iIJNS_1ZENS_1XENS_1YEEEEvv(
  template void i<Z, X, Y>();

  template<typename ...T> struct A {};
  template<typename, typename> struct B {};

  template<template<typename ...> typename> void p() {}
  // CHECK: define {{.*}}@_ZN5test51pINS_1AEEEvv(
  // CLANG16: define {{.*}}@_ZN5test51pINS_1AEEEvv(
  template void p<A>();
  // CHECK: define {{.*}}@_ZN5test51pITtTpTyENS_1BEEEvv(
  // CLANG16: define {{.*}}@_ZN5test51pINS_1BEEEvv(
  template void p<B>();

  template<template<typename, typename> typename> void q() {}
  // CHECK: define {{.*}}@_ZN5test51qITtTyTyENS_1AEEEvv(
  // CLANG16: define {{.*}}@_ZN5test51qINS_1AEEEvv(
  template void q<A>();
  // CHECK: define {{.*}}@_ZN5test51qINS_1BEEEvv(
  // CLANG16: define {{.*}}@_ZN5test51qINS_1BEEEvv(
  template void q<B>();
}

namespace test6 {
  // Abbreviated function templates.
  void f(C auto) {}
  // CHECK: define {{.*}}@_ZN5test61fITk1CiEEvT_(
  // CLANG16: define {{.*}}@_ZN5test61fIiEEvT_(
  template void f(int);

  template<typename T>
  void g(D<T> auto) {}
  // CHECK: define {{.*}}@_ZN5test61gIiTk1DIT_EiEEvT0_(
  // CLANG16: define {{.*}}@_ZN5test61gIiiEEvT0_(
  template void g<int>(int);
}

namespace test7 {
  // Constrained lambdas.
  template<typename T> void f() {
    // Ensure that requires-clauses affect lambda numbering.
    // CHECK-LABEL: define {{.*}}@_ZN5test71fIiEEvv(
    // CHECK: call {{.*}}@_ZZN5test71fIiEEvvENKUlTyQaa1CIT_E1CITL0__ET0_E_clIiiEEDaS3_Q1CIDtfp_EE(
    ([]<typename U> requires C<T> && C<U> (auto x) requires C<decltype(x)> {}).template operator()<int>(0);
    // CHECK: call {{.*}}@_ZZN5test71fIiEEvvENKUlTyQaa1CIT_E1CITL0__ET0_E0_clIiiEEDaS3_Qaa1CIDtfp_EELb1E(
    ([]<typename U> requires C<T> && C<U> (auto x) requires C<decltype(x)> && true {}).template operator()<int>(0);
    // CHECK: call {{.*}}@_ZZN5test71fIiEEvvENKUlTyQaa1CIT_E1CITL0__ET0_E1_clIiiEEDaS3_Q1CIDtfp_EE(
    ([]<typename U> requires C<T> && C<U> (auto x) requires C<decltype(x)> {}).template operator()<int>(0);
    // CHECK: call {{.*}}@_ZZN5test71fIiEEvvENKUlTyT0_E_clIiiEEDaS1_(
    ([]<typename U> (auto x){}).template operator()<int>(0);
  }
  template void f<int>();
}

namespace gh67244 {
  template<typename T, typename ...Ts> constexpr bool B = true;
  template<typename T, typename ...Ts> concept C = B<T, Ts...>;
  template<C<int, float> T> void f(T) {}
  // CHECK: define {{.*}} @_ZN7gh672441fITkNS_1CIifEEiEEvT_(
  template void f(int);
}

namespace gh67356 {
  template<typename, typename T> concept C = true;
  template<typename T> void f(T t, C<decltype(t)> auto) {}
  // CHECK: define {{.*}} @_ZN7gh673561fIiTkNS_1CIDtfL0p_EEEiEEvT_T0_(
  template void f(int, int);

  // Note, we use `fL0p` not `fp` above because:
  template<typename T> void g(T t, C<auto (T u) -> decltype(f(t, u))> auto) {}
  // CHECK: define {{.*}} @_ZN7gh673561gIiTkNS_1CIFDTcl1ffL0p_fp_EET_EEEiEEvS3_T0_(
  template void g(int, int);
}
