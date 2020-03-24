// RUN: %clang_cc1 -ftapir=serial -fsyntax-only -verify %s
#include<kitsune.h>

void f1() {
  int n;

  forall (n = 0; n < 10; n++);

  forall (n = 0 n < 10; n++); // expected-error {{expected ';' in 'for'}}
  forall (n = 0; n < 10 n++); // expected-error {{expected ';' in 'for'}}

  forall (int n = 0 n < 10; n++); // expected-error {{expected ';' in 'for'}}
  forall (int n = 0; n < 10 n++); // expected-error {{expected ';' in 'for'}}

  forall (n = 0 bool b = n < 10; n++); // expected-error {{expected ';' in 'for'}}
  forall (n = 0; bool b = n < 10 n++); // expected-error {{expected ';' in 'for'}}

  forall (n = 0 n < 10 n++); // expected-error 2{{expected ';' in 'for'}}

  forall (;); // expected-error {{expected ';' in 'for'}}
}
