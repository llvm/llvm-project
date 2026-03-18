// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s

a(   ::template operator // expected-error {{a type specifier is required for all declarations}} \
                         // expected-error {{expected ';' after top level declarator}}
// expected-error@* {{expected a type}}
// expected-error@* {{expected unqualified-id}}
