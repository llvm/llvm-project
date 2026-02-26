// RUN: %clang_cc1 -fsyntax-only -verify %s

void f(int);
int n = 5;

char *a = _Generic("", char (*)[f(x)]: ""); // expected-error {{use of undeclared identifier 'x'}} \
                                            // expected-error {{type 'char (*)[f(x)]' in generic association is a variably modified type}}

int b = _Generic(1, int[n]: 2, default: 3); // expected-error {{type 'int[n]' in generic association is a variably modified type}}

int c = _Generic(1, int[n]: 2, char[n]: 3, default: 4); // expected-error {{type 'int[n]' in generic association is a variably modified type}} \
                                                        // expected-error {{type 'char[n]' in generic association is a variably modified type}}
