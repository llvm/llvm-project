// RUN: %clang_cc1 -verify -std=c++20 %s

namespace GH61885 {
void similar() { // expected-note {{'similar' declared here}}
  if constexpr (similer<>) {} // expected-error {{use of undeclared identifier 'similer'; did you mean 'similar'?}} \
                                 expected-warning {{address of function 'similar<>' will always evaluate to 'true'}} \
                                 expected-note {{prefix with the address-of operator to silence this warning}}
}
void a() { if constexpr (__adl_swap<>) {}} // expected-error{{use of undeclared identifier '__adl_swap'}}

int AA() { return true;} // expected-note {{'AA' declared here}}

void b() { if constexpr (AAA<>) {}} // expected-error {{use of undeclared identifier 'AAA'; did you mean 'AA'?}} \
                                       expected-warning {{address of function 'AA<>' will always evaluate to 'true'}} \
                                       expected-note {{prefix with the address-of operator to silence this warning}}
}

