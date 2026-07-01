// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template <template <int> typename> struct S;
template <int, int = (foo<void, void>())> struct T; // expected-error {{use of undeclared identifier 'foo'}}
template <typename...> struct U;

using V = U<S<T>>;
