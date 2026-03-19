// RUN: %clang_cc1 -fsyntax-only -verify %s
a(   ::template operator // expected-error 2 {{expected a type}} \
                         // expected-error {{a type specifier is required for all declarations}} \
                         // expected-error {{expected ';' after top level declarator}}



