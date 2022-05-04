// RUN: %clang_cc1 -fsyntax-only -verify %s

// Don't crash.

template<typename>struct ae_same; // expected-note {{}}
template<typename>struct ts{}ap() // expected-error {{expected ';' after struct}} expected-error {{a type specifier is required}}
{ts<a>::ap<ae_same<int>::&ae_same<>>::p(a); }; // expected-error {{use of undeclared identifier 'a'}} expected-error 5{{}}
