// RUN: %clang_cc1 -std=c++2a -verify %s

int f(); // expected-note 2{{declared here}}

constinit int a;
constinit int b = f(); // expected-error {{does not have a constant initializer}} expected-note {{required by}} expected-note {{non-constexpr function 'f'}}
extern constinit int c; // expected-note {{here}} expected-note {{required by}}
int c = f(); // expected-warning {{missing}} expected-error {{does not have a constant initializer}} expected-note {{non-constexpr function 'f'}}
