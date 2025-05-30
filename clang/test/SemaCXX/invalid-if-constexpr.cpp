// RUN: %clang_cc1 -verify -std=c++20 %s

namespace GH61885 {
void similar() {
  if constexpr (similer<>) {} // expected-error {{use of undeclared identifier 'similer'}}
}
void a() { if constexpr (__adl_swap<>) {}} // expected-error{{use of undeclared identifier '__adl_swap'}}

int AA() { return true;}

void b() { if constexpr (AAA<>) {}} // expected-error {{use of undeclared identifier 'AAA'}}
}

