// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template <typename... T> requires ((sizeof(T) > 0) && ...) void f() {} // expected-note{{previous definition is here}}
class A;
void operator&&(A, A);
template <typename... T> requires ((sizeof(T) > 0) && ...) void f() {} // expected-error{{redefinition of 'f'}}
