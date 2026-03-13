// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -I %S/Inputs/redecl-templates %s -verify -std=c++14
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/redecl-templates %s -verify -std=c++14

template<int N> struct A {};
template<int N> using X = A<N>;

template<int N> constexpr void f() {}
template<int N> constexpr void g() { f<N>(); }

template<int N> extern int v;
template<int N> int &w = v<N>;

#include "a.h"

// Be careful not to mention A here, that'll import the decls from "a.h".
int g(X<1> *);
X<1> *p = 0;

// This will implicitly instantiate A<1> if we haven't imported the explicit
// specialization declaration from "a.h".
int k = g(p);
// Likewise for f and v.
void h() { g<1>(); }
int &x = w<1>;

// This is OK: we declared the explicit specialization before we triggered
// instantiation of this specialization.
template<> struct A<1> {};
template<> constexpr void f<1>() {}
// Variable template explicit specializations are always definitions unless they
// are static data members declared without an initializer.
template<> int v<1>; // expected-error {{redefinition of 'v<1>'}}
                     // expected-note@Inputs/redecl-templates/a.h:8 {{previous definition is here}}
