// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// GH188014
& decltype ( ( union { // expected-error {{expected ';' after union}} expected-error {{expected '}'}} expected-error {{expected ')'}} expected-error {{expected expression}} expected-error {{expected unqualified-id}} expected-note {{to match this '('}} expected-note {{to match this '{'}}
