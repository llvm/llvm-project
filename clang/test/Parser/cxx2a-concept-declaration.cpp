// Support parsing of concepts

// RUN:  %clang_cc1 -std=c++2a -fconcepts-ts -verify %s
template<typename T> concept C1 = true; // expected-note 2{{previous}}

template<typename T> concept C1 = true; // expected-error{{redefinition}}

template<concept T> concept D1 = true;
// expected-error@-1{{expected template parameter}}
// expected-error@-2{{concept template parameter list must have at least one parameter; explicit specialization of concepts is not allowed}}

template<template<typename> concept T> concept D2 = true;
// expected-error@-1{{expected identifier}}
// expected-error@-2{{template template parameter requires 'class' after the parameter list}}
// expected-error@-3{{concept template parameter list must have at least one parameter; explicit specialization of concepts is not allowed}}

template<typename T> concept C2 = 0.f; // expected-error {{constraint expression must be of type 'bool' but is of type 'float'}}

struct S1 {
  template<typename T> concept C1 = true; // expected-error {{concept declarations may only appear in global or namespace scope}}
};

extern "C++" {
  template<typename T> concept C1 = true; // expected-error{{redefinition}}
}

template<typename A>
template<typename B>
concept C4 = true; // expected-error {{extraneous template parameter list in concept definition}}

template<typename T> concept C5 = true; // expected-note {{previous}} expected-note {{previous}}
int C5; // expected-error {{redefinition}}
struct C5 {}; // expected-error {{redefinition}}

struct C6 {}; // expected-note{{previous definition is here}}
template<typename T> concept C6 = true;
// expected-error@-1{{redefinition of 'C6' as different kind of symbol}}

// TODO: Add test to prevent explicit specialization, partial specialization
// and explicit instantiation of concepts.

template<typename T, T v>
struct integral_constant { static constexpr T value = v; };

namespace N {
  template<typename T> concept C7 = true;
}
using N::C7;

template <bool word> concept C8 = integral_constant<bool, wor>::value;
// expected-error@-1{{use of undeclared identifier 'wor'; did you mean 'word'?}}
// expected-note@-2{{'word' declared here}}

template<typename T> concept bool C9 = true;
// expected-warning@-1{{ISO C++2a does not permit the 'bool' keyword after 'concept'}}

template<> concept C10 = false;
// expected-error@-1{{concept template parameter list must have at least one parameter; explicit specialization of concepts is not allowed}}

template<> concept C9<int> = false;
// expected-error@-1{{name defined in concept definition must be an identifier}}

template<typename T> concept N::C11 = false;
// expected-error@-1{{name defined in concept definition must be an identifier}}

class A { };
// expected-note@-1{{'A' declared here}}

template<typename T> concept A::C12 = false;
// expected-error@-1{{expected namespace name}}

template<typename T> concept operator int = false;
// expected-error@-1{{name defined in concept definition must be an identifier}}
