// RUN: %clang_cc1 -fsyntax-only -verify %s

void f(int);

char *a = _Generic("", char (*)[f(x)]: ""); // expected-error {{use of undeclared identifier 'x'}} \
                                            // expected-error {{type 'char (*)[f(x)]' in generic association is a variably modified type}}
