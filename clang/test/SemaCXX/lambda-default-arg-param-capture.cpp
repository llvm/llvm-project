// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

int a(int g, int h = []{ return g; }()); // expected-error {{variable 'g' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{'g' declared here}} expected-note {{lambda expression begins here}} expected-note {{capture 'g' by value}} expected-note {{capture 'g' by reference}} expected-note {{default capture by value}} expected-note {{default capture by reference}}

int e(int g, int h = [] { int g = 1; return g; }());

// Unevaluated operand in a trailing return type: still accepted.
auto u(int g) -> decltype([] { return g; }());

// A default argument is potentially evaluated even when the lambda sits in an
// unevaluated operand; GCC rejects these too.
int s1(int x, int h = sizeof([] { return x; }())); // expected-error {{variable 'x' cannot be implicitly captured in a lambda with no capture-default specified}} \
                                                      expected-note {{'x' declared here}} \
                                                      expected-note {{lambda expression begins here}} \
                                                      expected-note {{capture 'x' by value}} \
                                                      expected-note {{capture 'x' by reference}} \
                                                      expected-note {{default capture by value}} \
                                                      expected-note {{default capture by reference}}
