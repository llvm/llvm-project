// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

auto x0 = requires (this int) { true; }; // expected-error {{a requires expression cannot have an explicit object parameter}}
auto x1 = requires (int, this int) { true; }; // expected-error {{a requires expression cannot have an explicit object parameter}}

template<this auto>
void f(); // expected-error {{expected template parameter}}
          // expected-error@-1 {{no function template matches function template specialization 'f'}}
