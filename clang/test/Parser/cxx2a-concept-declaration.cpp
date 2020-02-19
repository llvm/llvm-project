// Support parsing of concepts

// RUN:  %clang_cc1 -std=c++20 -verify %s
template<typename T> concept C1 = true; // expected-note 2{{previous}}

template<typename T> concept C1 = true; // expected-error{{redefinition}}

template<concept T> concept D1 = true;
// expected-error@-1{{expected template parameter}}
// expected-error@-2{{concept template parameter list must have at least one parameter; explicit specialization of concepts is not allowed}}

template<template<typename> concept T> concept D2 = true;
// expected-error@-1{{expected identifier}}
// expected-error@-2{{template template parameter requires 'class' after the parameter list}}
// expected-error@-3{{concept template parameter list must have at least one parameter; explicit specialization of concepts is not allowed}}

struct S1 {
  template<typename T> concept C1 = true; // expected-error {{concept declarations may only appear in global or namespace scope}}
};

extern "C++" {
  template<typename T> concept C1 = true; // expected-error{{redefinition}}
}

template<typename A>
template<typename B>
concept C2 = true; // expected-error {{extraneous template parameter list in concept definition}}

template<typename T> concept C3 = true; // expected-note {{previous}} expected-note {{previous}}
int C3; // expected-error {{redefinition}}
struct C3 {}; // expected-error {{redefinition}}

struct C4 {}; // expected-note{{previous definition is here}}
template<typename T> concept C4 = true;
// expected-error@-1{{redefinition of 'C4' as different kind of symbol}}

// TODO: Add test to prevent explicit specialization, partial specialization
// and explicit instantiation of concepts.

template<typename T, T v>
struct integral_constant { static constexpr T value = v; };

namespace N {
  template<typename T> concept C5 = true;
}
using N::C5;

template <bool word> concept C6 = integral_constant<bool, wor>::value;
// expected-error@-1{{use of undeclared identifier 'wor'; did you mean 'word'?}}
// expected-note@-2{{'word' declared here}}

template<typename T> concept bool C7 = true;
// expected-warning@-1{{ISO C++20 does not permit the 'bool' keyword after 'concept'}}

template<> concept C8 = false;
// expected-error@-1{{concept template parameter list must have at least one parameter; explicit specialization of concepts is not allowed}}

template<> concept C7<int> = false;
// expected-error@-1{{name defined in concept definition must be an identifier}}

template<typename T> concept N::C9 = false;
// expected-error@-1{{name defined in concept definition must be an identifier}}

class A { };
// expected-note@-1{{'A' declared here}}

template<typename T> concept A::C10 = false;
// expected-error@-1{{expected namespace name}}

template<typename T> concept operator int = false;
// expected-error@-1{{name defined in concept definition must be an identifier}}

template<bool x> concept C11 = 2; // expected-error {{atomic constraint must be of type 'bool' (found 'int')}}
template<bool x> concept C12 = 2 && x; // expected-error {{atomic constraint must be of type 'bool' (found 'int')}}
template<bool x> concept C13 = x || 2 || x; // expected-error {{atomic constraint must be of type 'bool' (found 'int')}}
template<bool x> concept C14 = 8ull && x || x; // expected-error {{atomic constraint must be of type 'bool' (found 'unsigned long long')}}
template<typename T> concept C15 = sizeof(T); // expected-error {{atomic constraint must be of type 'bool'}}
template<typename T> concept C16 = true && (0 && 0); // expected-error {{atomic constraint must be of type 'bool' (found 'int')}}
// expected-warning@-1{{use of logical '&&' with constant operand}}
// expected-note@-2{{use '&' for a bitwise operation}}
// expected-note@-3{{remove constant to silence this warning}}
template<typename T> concept C17 = T{};
static_assert(!C17<bool>);
template<typename T> concept C18 = (bool&&)true;
static_assert(C18<int>);
template<typename T> concept C19 = (const bool&)true;
static_assert(C19<int>);
template<typename T> concept C20 = (const bool)true;
static_assert(C20<int>);
template <bool c> concept C21 = integral_constant<bool, c>::value && true;
static_assert(C21<true>);
static_assert(!C21<false>);
template <bool c> concept C22 = integral_constant<bool, c>::value;
static_assert(C22<true>);
static_assert(!C22<false>);

template <bool word> concept C23 = integral_constant<bool, wor>::value;
// expected-error@-1{{use of undeclared identifier 'wor'; did you mean 'word'?}}
// expected-note@-2{{'word' declared here}}

