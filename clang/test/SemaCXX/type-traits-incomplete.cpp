// RUN: %clang_cc1 -fsyntax-only -verify %s 

struct S; // expected-note 2 {{forward declaration of 'S'}}

void f() {
  __is_pod(S); // expected-error{{incomplete type 'S' used in type trait expression}}
  __is_pod(S[]); // expected-error{{incomplete type 'S' used in type trait expression}}
}
