// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s

a(   ::template operator // expected-error {{expected a type}} expected-error {{unknown type name 'a'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected unqualified-id}}
