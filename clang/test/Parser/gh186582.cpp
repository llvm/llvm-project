// RUN: %clang_cc1 -fsyntax-only -verify %s

a(   ::template operator
// expected-error@-1 {{unknown type name 'a'}}
// expected-error@-2 {{expected unqualified-id}}
// expected-error@-3 {{expected ')'}}
// expected-note@-4 {{to match this '('}}
// expected-error@* 2{{expected a type}}
