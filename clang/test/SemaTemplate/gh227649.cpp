// RUN: %clang_cc1 -fsyntax-only -verify %s

template <template <decltype(foo())> typename> struct S {}; // expected-error {{use of undeclared identifier 'foo'}}
template <int &p> struct P;
S<P> s;
