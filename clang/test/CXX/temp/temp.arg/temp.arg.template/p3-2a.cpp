// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T> concept C = T::f();
// expected-note@-1{{similar constraint}}
template<typename T> concept D = C<T> && T::g();
template<typename T> concept F = T::f();
// expected-note@-1{{similar constraint expressions not considered equivalent}}
template<template<C> class P> struct S1 { }; // expected-note 2{{'P' declared here}}

template<C> struct X { };

template<D> struct Y { }; // expected-note{{'Y' declared here}}
template<typename T> struct Z { };
template<F> struct W { }; // expected-note{{'W' declared here}}

S1<X> s11;
S1<Y> s12; // expected-error{{template template argument 'Y' is more constrained than template template parameter 'P'}}
S1<Z> s13;
S1<W> s14; // expected-error{{template template argument 'W' is more constrained than template template parameter 'P'}}

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
