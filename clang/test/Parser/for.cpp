// RUN: %clang_cc1 -fsyntax-only -verify %s

void f1() {
  int n;

  for (n = 0; n < 10; n++);

  for (n = 0 n < 10; n++); // expected-error {{expected ';' in 'for'}}
  for (n = 0; n < 10 n++); // expected-error {{expected ';' in 'for'}}

  for (int n = 0 n < 10; n++); // expected-error {{expected ';' in 'for'}}
  for (int n = 0; n < 10 n++); // expected-error {{expected ';' in 'for'}}

  for (n = 0 bool b = n < 10; n++); // expected-error {{expected ';' in 'for'}}
  for (n = 0; bool b = n < 10 n++); // expected-error {{expected ';' in 'for'}}

  for (n = 0 n < 10 n++); // expected-error 2{{expected ';' in 'for'}}

  for (;); // expected-error {{expected ';' in 'for'}}
}
