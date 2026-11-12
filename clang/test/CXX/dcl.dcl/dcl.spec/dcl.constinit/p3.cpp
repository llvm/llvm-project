// RUN: %clang_cc1 -std=c++2a -verify %s

const char *g() { return "dynamic initialization"; } // expected-note {{declared here}}
constexpr const char *f(bool b) { return b ? "constant initialization" : g(); } // expected-note {{non-constexpr function 'g'}}
constinit const char *c = f(true);
constinit const char *d = f(false); // expected-error {{does not have a constant initializer}} expected-note 2{{}}
