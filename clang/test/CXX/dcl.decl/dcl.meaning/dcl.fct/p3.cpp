// RUN: %clang_cc1 -fsyntax-only -verify %s 
void f(int) { } // expected-note {{previous definition is here}}
void f(const int) { } // expected-error {{redefinition of 'f'}}

template <typename T> void ft(T) {}
template <> void ft(int) {} // expected-note {{previous definition is here}}
template <> void ft(int) {} // expected-error {{redefinition of 'ft<int>'}}
