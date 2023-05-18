// RUN: %clang_cc1 -verify %s

namespace deduce_pack_non_pack {
  template <typename...> class A;
  template <typename> struct C {};
  template <typename T> void g(C<A<T>>); // expected-note {{candidate template ignored: deduced type 'C<A<[...], (no argument)>>' of 1st parameter does not match adjusted type 'C<A<[...], int>>' of argument [with T = bool]}}
  void h(C<A<bool, int>> &x) { g(x); } // expected-error {{no matching function}}
}

namespace pr39231 {
  template<typename T, T ...V> struct integer_sequence {};

  template <typename T, T... A, T... B>
  int operator^(integer_sequence<T, A...> a, // expected-note {{deduced conflicting values for parameter 'A' (<1, 2, 3> vs. <4, 5, 6>)}}
                integer_sequence<T, A...> b);

  int v = integer_sequence<int, 1, 2, 3>{} ^ integer_sequence<int, 4, 5, 6>{}; // expected-error {{invalid operands}}

  template <typename T, T... A, T... B>
  integer_sequence<T, A + B...> operator+(integer_sequence<T, A...> a,
                                          integer_sequence<T, B...> b);
  integer_sequence<int, 5, 7, 9> w =
      integer_sequence<int, 1, 2, 3>{} + integer_sequence<int, 4, 5, 6>{};
}
