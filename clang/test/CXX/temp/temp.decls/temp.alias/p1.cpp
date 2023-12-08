// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T> using U = T;

// The name of the alias template is a template-name.
U<char> x;
void f(U<int>);
typedef U<U<U<U<int>>>> I;
