// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

void f();

// Test typeof(expr) canonicalization
template<typename T, T N>
void f0(T x, decltype(f(N, x)) y) { } // expected-note{{previous}}

template<typename T, T N>
void f0(T x, decltype((f)(N, x)) y) { }

template<typename U, U M>
void f0(U u, decltype(f(M, u))) { } // expected-error{{redefinition}}

// PR12438: Test sizeof...() canonicalization
template<int> struct N {};

template<typename...T>
N<sizeof...(T)> f1() {} // expected-note{{previous}}

template<typename, typename...T>
N<sizeof...(T)> f1() {}

template<class...U>
N<sizeof...(U)> f1() {} // expected-error{{redefinition}}
