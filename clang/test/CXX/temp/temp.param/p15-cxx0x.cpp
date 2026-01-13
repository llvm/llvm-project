// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
template <typename T> struct X; // expected-note {{'X<X<X<int>>>' is incomplete}}
template<int I> struct Y;

X<X<int>> *x1;

Y<(1 >> 2)> *y1;
Y<1 >> 2> *y2; // FIXME: expected-error{{expected unqualified-id}}

X<X<X<X<X<int>>>>> *x2;

template<> struct X<int> { };
typedef X<int> X_int;
struct Z : X_int { };

void f(const X<int> x) {
  (void)reinterpret_cast<X<int>>(x);       // expected-error{{reinterpret_cast from 'const X<int>' to 'X<int>' is not allowed}}
  (void)reinterpret_cast<X<X<X<int>>>>(x); // expected-error{{reinterpret_cast from 'const X<int>' to 'X<X<X<int>>>' is not allowed}}

  X<X<int>> *x1;
}

template<typename T = void> struct X1 { };
X1<X1<>> x1a;


namespace ParameterPackExpansions {

// A template parameter pack that [contains an unexpanded parameter pack] is a
// pack expansion.

template<typename...Ts> struct Outer {
  // From [temp.variadic]p4:
  //   In a template parameter pack that is a pack expansion, the pattern is
  //   [...the template-parameter...] without the ellipsis.
  // Therefore the resulting sequence of parameters is not a parameter pack,
  // so is not required to be the last template parameter.
  template<Ts ...As, template<Ts> class ...Bs, typename ...Cs> struct Inner {
    struct Check : Bs<As>... {
      Check(Cs...);
    };
  };
};

template<int> struct TemplateInt {};
template<char> struct TemplateChar {};
template<int*> struct TemplateIntPtr {};
int x;

Outer<int, char, int*>::
Inner<12345, 'x', &x,
      TemplateInt, TemplateChar, TemplateIntPtr,
      int*>::
Check check(&x);


template<typename...Ts> struct types;

enum place { _ };
template<place...> struct places {};

template<typename P1, typename P2> struct append_places;
template<place...X1, place...X2>
struct append_places<places<X1...>, places<X2...>> {
  typedef places<X1...,X2...> type;
};

template<unsigned N>
struct make_places : append_places<typename make_places<N/2>::type,
                                   typename make_places<N-N/2>::type> {};
template<> struct make_places<0> { typedef places<> type; };
template<> struct make_places<1> { typedef places<_> type; };

template<typename T> struct wrap {
  template<place> struct inner { typedef T type; };
};

template<typename T> struct takedrop_impl;
template<place...X> struct takedrop_impl<places<X...>> {
  template<template<decltype(X)> class ...Take,
           template<place      > class ...Drop>
  struct inner { // expected-note 2{{declared}}
    typedef types<typename Take<_>::type...> take;
    typedef types<typename Drop<_>::type...> drop;
  };
};

template<unsigned N, typename...Ts> struct take {
  using type = typename takedrop_impl<typename make_places<N>::type>::
    template inner<wrap<Ts>::template inner...>::take; // expected-error {{too few template arguments}}
};
template<unsigned N, typename...Ts> struct drop {
  using type = typename takedrop_impl<typename make_places<N>::type>::
    template inner<wrap<Ts>::template inner...>::drop; // expected-error {{too few template arguments}}
};

using T1 = take<3, int, char, double, long>::type; // expected-note {{previous}}
// FIXME: Desguar the types on the RHS in this diagnostic.
// desired-error {{'types<void, void, void, void>' vs 'types<int, char, double, (no argument)>'}}
using T1 = types<void, void, void, void>; // expected-error {{'types<void, void, void, void>' vs 'types<typename ParameterPackExpansions::wrap<int>::inner<_>::type, typename ParameterPackExpansions::wrap<char>::inner<_>::type, typename ParameterPackExpansions::wrap<double>::inner<_>::type, (no argument)>'}}
using D1 = drop<3, int, char, double, long>::type;
using D1 = types<long>;

using T2 = take<4, int, char, double, long>::type; // expected-note {{previous}}
// FIXME: Desguar the types on the RHS in this diagnostic.
// desired-error {{'types<void, void, void, void>' vs 'types<int, char, double, long>'}}
using T2 = types<void, void, void, void>; // expected-error {{'types<void, void, void, void>' vs 'types<typename ParameterPackExpansions::wrap<int>::inner<_>::type, typename ParameterPackExpansions::wrap<char>::inner<_>::type, typename ParameterPackExpansions::wrap<double>::inner<_>::type, typename ParameterPackExpansions::wrap<long>::inner<_>::type>'}}
using T2 = types<int, char, double, long>;
using D2 = drop<4, int, char, double, long>::type;
using D2 = types<>;

using T3 = take<5, int, char, double, long>::type; // expected-note {{in instantiation of}}
using D3 = drop<5, int, char, double, long>::type; // expected-note {{in instantiation of}}


// FIXME: We should accept this code. A parameter pack within a default argument
// in a template template parameter pack is expanded, because the pack is
// implicitly a pack expansion.
template<typename ...Default> struct DefArg {
  template<template<typename T = Default> class ...Classes> struct Inner { // expected-error {{default argument contains unexpanded parameter pack}} expected-note {{here}}
    Inner(Classes<>...); // expected-error {{too few}}
  };
};
template<typename T> struct vector {};
template<typename T> struct list {};
vector<int> vi;
list<char> lc;
DefArg<int, char>::Inner<vector, list> defarg(vi, lc);


// FIXME:
// A template parameter pack that is a pack expansion shall not expand a
// parameter pack declared in the same template-parameter-list.
template<typename...Ts, Ts...Vs> void error(); // desired-error

// This case should not produce an error, because in A's instantiation, Cs is
// not a parameter pack.
template<typename...Ts> void consume(Ts...);
template<typename...Ts> struct A {
  template<template<typename, Ts = 0> class ...Cs, Cs<Ts> ...Vs> struct B { // ok
    B() {
      consume([]{
        int arr[Vs]; // expected-error {{negative size}}
                     // expected-note@-2 {{while substituting into a lambda expression here}}
      }...);
    }
  };
};
template<typename, int> using Int = int;
template<typename, short> using Char = char;
A<int, short>::B<Int, Char, -1, 'x'> b; // expected-note {{here}}

}

namespace PR9023 {
  template<typename ...T> struct A {
    template<template<T> class ...> struct B {
    };
  };

  template<int> struct C { };
  template<long> struct D { };

  int main() {
    A<int, long>::B<C, D> e;
  }
}

namespace std_examples {
  template <class... Types> class Tuple;
  template <class T, int... Dims> struct multi_array;
  template <class... T> struct value_holder {
    template<T... Values> struct apply { };
  };
  template <class... T, T... Values> struct static_array; // expected-error {{must be the last}}

  int n;
  value_holder<int, char, int*>::apply<12345, 'x', &n> test;
}
