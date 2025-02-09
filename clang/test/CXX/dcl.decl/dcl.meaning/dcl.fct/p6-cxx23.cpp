// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

auto x0 = requires (this int) { true; }; // expected-error {{a requires expression cannot have an explicit object parameter}}
auto x1 = requires (int, this int) { true; }; // expected-error {{a requires expression cannot have an explicit object parameter}}

template<this auto> // expected-error {{expected template parameter}}
void f(); // expected-error {{no function template matches function template specialization 'f'}}
