// RUN: %clang_cc1 -std=c++1z -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

template<int> struct A {};
template<int ...N> void foldr(A<(N + ...)>);
template<int ...N> void foldl(A<(... + N)>);
template<int ...N> void foldr1(A<(N + ... + 1)>);
template<int ...N> void foldl1(A<(1 + ... + N)>);
void use() {
  foldr<1, 2, 3>({});
  foldl<1, 2, 3>({});
  foldr1<1, 2, 3>({});
  foldl1<1, 2, 3>({});
  // CHECK-DAG: @_Z5foldrIJLi1ELi2ELi3EEEv1AIXfrplT_EE(
  // CHECK-DAG: @_Z5foldlIJLi1ELi2ELi3EEEv1AIXflplT_EE(
  // CHECK-DAG: @_Z6foldr1IJLi1ELi2ELi3EEEv1AIXfRplT_Li1EEE(
  // CHECK-DAG: @_Z6foldl1IJLi1ELi2ELi3EEEv1AIXfLplLi1ET_EE(
}

template<int ...N> using Foldr = A<(N + ...)>;
template<int ...N> using Foldl = A<(... + N)>;
template<int ...N> using Foldr1 = A<(N + ... + 1)>;
template<int ...N> using Foldl1 = A<(1 + ... + N)>;

template<int ...A> struct Partial {
  template<int ...B> void foldr(Foldr<A..., B..., A..., B...>);
  template<int ...B> void foldl(Foldl<A..., B..., A..., B...>);
  template<int ...B> void foldr1(Foldr1<A..., B..., A..., B...>);
  template<int ...B> void foldl1(Foldl1<A..., B..., A..., B...>);
};
void use(Partial<1, 2> p) {
  p.foldr<3, 4>({});
  p.foldl<3, 4>({});
  p.foldr1<3, 4>({});
  p.foldl1<3, 4>({});
  // CHECK-DAG: @_ZN7PartialIJLi1ELi2EEE5foldrIJLi3ELi4EEEEv1AIXplLi1EplLi2EfRplT_plLi1EplLi2EfrplT_EE(
  // CHECK-DAG: @_ZN7PartialIJLi1ELi2EEE5foldlIJLi3ELi4EEEEv1AIXfLplplplfLplplLi1ELi2ET_Li1ELi2ET_EE
  // CHECK-DAG: @_ZN7PartialIJLi1ELi2EEE6foldr1IJLi3ELi4EEEEv1AIXplLi1EplLi2EfRplT_plLi1EplLi2EfRplT_Li1EEE(
  // CHECK-DAG: @_ZN7PartialIJLi1ELi2EEE6foldl1IJLi3ELi4EEEEv1AIXfLplplplfLplplplLi1ELi1ELi2ET_Li1ELi2ET_EE(
}

extern int n;
template<int ...N> void f() {
  (n = ... = N);
}
template void f<>();
