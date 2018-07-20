// RUN: %clang_cc1 -verify %s

namespace deduce_pack_non_pack {
  template <typename...> class A;
  template <typename> struct C {};
  template <typename T> void g(C<A<T>>); // expected-note {{candidate template ignored: deduced type 'C<A<[...], (no argument)>>' of 1st parameter does not match adjusted type 'C<A<[...], int>>' of argument [with T = bool]}}
  void h(C<A<bool, int>> &x) { g(x); } // expected-error {{no matching function}}
}
