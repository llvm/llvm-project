// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions -fexceptions -fcxx-exceptions

void f() throw(...) { }

namespace PR28080 {
struct S;           // expected-note {{forward declaration}}
void fn() throw(S); // expected-warning {{incomplete type}} expected-note{{previous declaration}}
void fn() throw();  // expected-warning {{does not match previous declaration}}
}
