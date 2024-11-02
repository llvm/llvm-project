// RUN:  %clang_cc1 -std=c++2a -frelaxed-template-template-args -verify %s

template<typename T> concept C = T::f(); // #C
template<typename T> concept D = C<T> && T::g();
template<typename T> concept F = T::f(); // #F
template<template<C> class P> struct S1 { }; // #S1

template<C> struct X { };

template<D> struct Y { }; // #Y
template<typename T> struct Z { };
template<F> struct W { }; // #W
S1<X> s11;
S1<Y> s12;
// expected-error@-1 {{template template argument 'Y' is more constrained than template template parameter 'P'}}
// expected-note@#S1 {{'P' declared here}}
// expected-note@#Y {{'Y' declared here}}
S1<Z> s13;
S1<W> s14;
// expected-error@-1 {{template template argument 'W' is more constrained than template template parameter 'P'}}
// expected-note@#S1 {{'P' declared here}}
// expected-note@#W {{'W' declared here}}
// expected-note@#F 1-2{{similar constraint expressions not considered equivalent}}
// expected-note@#C 1-2{{similar constraint}}

template<template<typename> class P> struct S2 { };

S2<X> s21;
S2<Y> s22;
S2<Z> s23;

template <template <typename...> class C>
struct S3;

template <C T>
using N = typename T::type;

using s31 = S3<N>;
using s32 = S3<Z>;

template<template<typename T> requires C<T> class P> struct S4 { }; // #S4

S4<X> s41;
S4<Y> s42;
// expected-error@-1 {{template template argument 'Y' is more constrained than template template parameter 'P'}}
// expected-note@#S4 {{'P' declared here}}
// expected-note@#Y {{'Y' declared here}}
S4<Z> s43;
S4<W> s44;
// expected-error@-1 {{template template argument 'W' is more constrained than template template parameter 'P'}}
// expected-note@#S4 {{'P' declared here}}
// expected-note@#W {{'W' declared here}}

template<template<typename T> requires C<T> typename U> struct S5 {
  template<typename T> static U<T> V;
};

struct Nothing {};

// FIXME: Wait the standard to clarify the intent.
template<> template<> Z<Nothing> S5<Z>::V<Nothing>;
