// RUN: %clang_cc1 -fsyntax-only -verify %s
template <class ...>
struct Types;

// We check for negative condition and an error to make sure we don't accidentally produce a dependent expression.
template <class>
struct CheckSimpleTypes {
  static_assert(!__is_same( // expected-error {{static assertion failed}}
    Types<__builtin_sort_pack<int, double, int, double>...>,
    Types<double, double, int, int>)); 
};
template struct CheckSimpleTypes<int>; // expected-note {{requested here}}

__builtin_sort_pack<int, double> err1; // expected-error {{builtin returning packs used outside of template}} \
                                       // expected-error {{declaration type contains an unexpanded parameter pack}}
Types<__builtin_sort_pack<int, double>*> err2; // expected-error {{builtin returning packs used outside of template}} \
                                               // expected-error {{declaration type contains an unexpanded parameter pack}}
Types<const __builtin_sort_pack<int, double>> *err3; // expected-error {{builtin returning packs used outside of template}} \
                                                     // expected-error {{declaration type contains an unexpanded parameter pack}}

template <template <class...> class Inner>
struct Wrapper {
  using result = Inner<int,int,int>*;
};
Types<Wrapper<__builtin_sort_pack>::result> *err11;  // expected-error {{builtin returning packs used outside of template}} \
                                                     // expected-error  {{use of template '__builtin_sort_pack' requires template arguments}} \
                                                     // expected-note@* {{template declaration from hidden source}}
// Check we properly defer evaluations for dependent inputs.
// It is important for performance to avoid repeated sorting (mangling is expensive).
template <class T>
struct CheckDependent {
  template <class U>
  struct Inner {
    using S1 = Types<__builtin_sort_pack<T, U>...>;
    using S2 = Types<__builtin_sort_pack<U, T>...>;
    using S3 = Types<__builtin_sort_pack<double, T>...>;
    using S4 = Types<__builtin_sort_pack<U, int>...>;
  };
};

using IntDouble = CheckDependent<int>::Inner<double>;
static_assert(!__is_same(Types<double, int>, CheckDependent<int>::Inner<double>::S1));  //expected-error {{static assertion failed}}
static_assert(!__is_same(Types<double, int>, CheckDependent<int>::Inner<double>::S2));  //expected-error {{static assertion failed}}
static_assert(!__is_same(Types<double, int>, CheckDependent<int>::Inner<double>::S3));  //expected-error {{static assertion failed}}
static_assert(!__is_same(Types<double, int>, CheckDependent<int>::Inner<double>::S4));  //expected-error {{static assertion failed}}
