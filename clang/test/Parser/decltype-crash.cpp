// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace GH114815{
  #define ID(X) X
  extern int ID(decltype);
  // expected-error@-1 {{expected '(' after 'decltype'}}
  // expected-error@-2 {{expected unqualified-id}}
}

namespace GH165246 {
  int decltype {};
  // expected-error@-1 {{expected '(' after 'decltype'}}
  // expected-error@-2 {{expected unqualified-id}}
}

namespace GH211207 {
  int decltype = 0;
  // expected-error@-1 {{expected '(' after 'decltype'}}
  // expected-error@-2 {{expected unqualified-id}}

  int *decltype = 0;
  // expected-error@-1 {{expected '(' after 'decltype'}}
  // expected-error@-2 {{expected unqualified-id}}
}

// GH188014
& decltype ( ( union { // expected-error {{expected ';' after union}} expected-error {{expected '}'}} expected-error {{expected ')'}} expected-error {{expected expression}} expected-error {{expected unqualified-id}} expected-note {{to match this '('}} expected-note {{to match this '{'}}
