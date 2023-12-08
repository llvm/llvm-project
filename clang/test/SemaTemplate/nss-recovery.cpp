// RUN: %clang_cc1 -fsyntax-only -verify %s

// There should be no extra errors about missing 'typename' keywords.
void f() {
  decltype(undef())::Type T; // expected-error {{use of undeclared identifier}}
}
