// RUN: %clang_cc1 -verify -std=c++2a -fclang-abi-compat=latest -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s
// expected-no-diagnostics

template <typename T, int N> concept SmallerThan = sizeof(T) < N;
template <typename T> concept Small = SmallerThan<T, 1000>;

template <typename T> struct X { using type = T; };

template <typename T> void f(int n) requires requires {
  // simple-requirement
  T();
  n;
  n == T();
  // compound-requirement
  {T() + 1} -> Small;
  {T() - 1} noexcept;
  {T() * 2} noexcept -> SmallerThan<1234>;
  // type-requirement
  typename T;
  typename X<T>;
  typename X<T>::type;
  typename X<decltype(n)>;
  // nested-requirement
  requires SmallerThan<T, 256>;
} {}
// CHECK: define {{.*}}@_Z1fIiEviQrqXcvT__EXfp_Xeqfp_cvS0__EXplcvS0__ELi1ER5SmallXmicvS0__ELi1ENXmlcvS0__ELi2ENR11SmallerThanILi1234EETS0_T1XIS0_ETNS3_4typeETS2_IiEQ11SmallerThanIS0_Li256EEE(
template void f<int>(int);

template <typename T> void g(int n) requires requires (T m) {
  // reference to our parameter vs an enclosing parameter
  n + m;
} {}
// CHECK: define {{.*}}@_Z1gIiEviQrQT__XplfL0p_fp_E(
template void g<int>(int);
