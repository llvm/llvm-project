// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

auto x0 = requires (this int) { true; }; // expected-error {{a requires clause cannot have an explicit object parameter}}
auto x1 = requires (int, this int) { true; }; // expected-error {{a requires clause cannot have an explicit object parameter}}
