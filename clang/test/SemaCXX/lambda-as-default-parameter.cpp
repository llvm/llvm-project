// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct a; // expected-note {{forward declaration of 'a'}} \
             expected-note {{forward declaration of 'a'}}
void b(a c = [] { return c; }); // expected-error {{initialization of incomplete type 'a'}} \
                                   expected-error {{variable has incomplete type 'a'}}
