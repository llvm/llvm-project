// RUN:  %clang_cc1 -std=c++2a -verify -triple x86_64-linux-gnu %s

template<typename T> concept C1 = true; // expected-note{{template is declared here}}
static_assert(C1<int>);
static_assert(C1);
// expected-error@-1{{use of concept 'C1' requires template arguments}}

template<typename T> concept C2 = sizeof(T) == 4;
static_assert(C2<int>);
static_assert(!C2<long long int>);
static_assert(C2<char[4]>);
static_assert(!C2<char[5]>);

template<typename T> concept C3 = sizeof(*T{}) == 4;
static_assert(C3<int*>);
static_assert(!C3<long long int>);

struct A {
  static constexpr int add(int a, int b) {
    return a + b;
  }
};
struct B {
  static int add(int a, int b) { // expected-note{{declared here}}
    return a + b;
  }
};
template<typename U>
concept C4 = U::add(1, 2) == 3;
// expected-error@-1{{substitution into constraint expression resulted in a non-constant expression}}
// expected-note@-2{{non-constexpr function 'add' cannot be used in a constant expression}}
static_assert(C4<A>);
static_assert(!C4<B>); // expected-note {{while checking the satisfaction of concept 'C4<B>' requested here}}

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

template<typename T, typename U>
concept Same = is_same_v<T, U>;

static_assert(Same<int, int>);
static_assert(Same<int, decltype(1)>);
static_assert(!Same<int, unsigned int>);
static_assert(!Same<A, B>);
static_assert(Same<A, A>);

static_assert(Same<bool, decltype(C1<int>)>);
static_assert(Same<bool, decltype(C2<int>)>);
static_assert(Same<bool, decltype(C3<int*>)>);
static_assert(Same<bool, decltype(C4<A>)>);

template<typename T> concept C5 = T{}; // expected-error {{atomic constraint must be of type 'bool' (found 'int')}}
constexpr bool x = C5<int>; // expected-note {{while checking the satisfaction of concept 'C5<int>' requested here}}

template<int x>
concept IsEven = (x % 2) == 0;

static_assert(IsEven<20>);
static_assert(!IsEven<11>);

template<template<typename T> typename P>
concept IsTypePredicate = is_same_v<decltype(P<bool>::value), const bool>
                          && is_same_v<decltype(P<int>::value), const bool>
                          && is_same_v<decltype(P<long long>::value), const bool>;

template<typename T> struct T1 {};
template<typename T> struct T2 { static constexpr bool value = sizeof(T) == 2; };

static_assert(IsTypePredicate<T2>);
static_assert(!IsTypePredicate<T1>);

template<typename T, typename U, typename... Ts>
concept OneOf = (Same<T, Ts> || ...);

static_assert(OneOf<int, long, int>);
static_assert(!OneOf<long, int, char, char>);

namespace piecewise_substitution {
  template <typename T>
  concept True = true;

  template <typename T>
  concept A = True<T> || T::value;

  template <typename T>
  concept B = (True<T> || T::value);

  template <typename T>
  concept C = !True<T> && T::value || true;

  template <typename T>
  concept D = (!True<T> && T::value) || true;

  template <typename T>
  concept E = T::value || True<T>;

  template <typename T>
  concept F = (T::value || True<T>);

  template <typename T>
  concept G = T::value && !True<T> || true;

  template <typename T>
  concept H = (T::value && !True<T>) || true;

  template <typename T>
  concept I = T::value;

  static_assert(A<int>);
  static_assert(B<int>);
  static_assert(C<int>);
  static_assert(D<int>);
  static_assert(E<int>);
  static_assert(F<int>);
  static_assert(G<int>);
  static_assert(H<int>);
  static_assert(!I<int>);
}

// Short ciruiting

template<typename T> struct T3 { using type = typename T::type; };
// expected-error@-1{{type 'char' cannot be used prior to '::' because it has no members}}
// expected-error@-2{{type 'short' cannot be used prior to '::' because it has no members}}

template<typename T>
concept C6 = sizeof(T) == 1 && sizeof(typename T3<T>::type) == 1;
// expected-note@-1{{while substituting template arguments into constraint expression here}}
// expected-note@-2{{in instantiation of template class 'T3<char>' requested here}}

template<typename T>
concept C7 = sizeof(T) == 1 || sizeof(
// expected-note@-1{{while substituting template arguments into constraint expression here}}
    typename
      T3<T>
// expected-note@-1{{in instantiation of template class 'T3<short>' requested here}}
        ::type) == 1;

static_assert(!C6<short>);
static_assert(!C6<char>); // expected-note{{while checking the satisfaction of concept 'C6<char>' requested here}}
static_assert(C7<char>);
static_assert(!C7<short>); // expected-note{{while checking the satisfaction of concept 'C7<short>' requested here}}

// Make sure argument list is converted when instantiating a CSE.

template<typename T, typename U = int>
concept SameSize = sizeof(T) == sizeof(U);

template<typename T>
struct X { static constexpr bool a = SameSize<T>; };

static_assert(X<unsigned>::a);

// static_assert concept diagnostics
template<typename T>
concept Large = sizeof(T) > 100;
// expected-note@-1 2{{because 'sizeof(small) > 100' (1 > 100) evaluated to false}}

struct small { };
static_assert(Large<small>);
// expected-error@-1 {{static assertion failed}}
// expected-note@-2 {{because 'small' does not satisfy 'Large'}}
static_assert(Large<small>, "small isn't large");
// expected-error@-1 {{static assertion failed: small isn't large}}
// expected-note@-2 {{because 'small' does not satisfy 'Large'}}

// Make sure access-checking can fail a concept specialization

class T4 { static constexpr bool f = true; };
template<typename T> concept AccessPrivate = T{}.f;
// expected-note@-1{{because substituted constraint expression is ill-formed: 'f' is a private member of 'T4'}}
static_assert(AccessPrivate<T4>);
// expected-error@-1{{static assertion failed}}
// expected-note@-2{{because 'T4' does not satisfy 'AccessPrivate'}}

template<typename T, typename U>
// expected-note@-1{{template parameter is declared here}}
concept C8 = sizeof(T) > sizeof(U);

template<typename... T>
constexpr bool B8 = C8<T...>;
// expected-error@-1{{pack expansion used as argument for non-pack parameter of concept}}


// Make sure we correctly check for containsUnexpandedParameterPack

template<typename T>
concept C9 = true;

template <typename Fn, typename... Args>
using invoke = typename Fn::template invoke<Args...>;

template <typename C, typename... L>
// The converted argument here will not containsUnexpandedParameterPack, but the
// as-written one will.
requires (C9<invoke<C, L>> &&...)
struct S { };
