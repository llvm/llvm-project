// RUN: %clang_cc1 -verify -std=c++20 %s

namespace GH61885 {
void similar() { // expected-note {{'similar' declared here}}
  if constexpr (similer<>) {} // expected-error {{use of undeclared identifier 'similer'; did you mean 'similar'?}}
}
void a() { if constexpr (__adl_swap<>) {}} // expected-error{{use of undeclared identifier '__adl_swap'; did you mean '__sync_swap'?}}

int AA() { return true;} // expected-note {{'AA' declared here}}

void b() { if constexpr (AAA<>) {}} // expected-error {{use of undeclared identifier 'AAA'; did you mean 'AA'?}}
}

