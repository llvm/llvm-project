// RUN: %clang_cc1 -fsyntax-only -verify %s

int ab[7];
int[(int)(abc<>)]; // expected-error {{use of undeclared identifier 'abc'}} \
                   // expected-error {{expected expression}}

int[(int)(undefined_name<>)]; // expected-error {{use of undeclared identifier 'undefined_name'}} \
                              // expected-error {{expected expression}}

int[(int)(<>)]; // expected-error {{expected expression}}

int[(int)(123<>)]; // expected-error {{expected expression}}
