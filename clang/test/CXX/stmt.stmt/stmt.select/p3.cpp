// RUN: %clang_cc1 -fsyntax-only -verify %s

int f();

void g() {
  if (int x = f()) { // expected-note 2{{previous definition}}
    int x; // expected-error{{redefinition of 'x'}}
  } else {
    int x; // expected-error{{redefinition of 'x'}}
  }
}


void h() {
  if (int x = f()) // expected-note 2{{previous definition}}
    int x; // expected-error{{redefinition of 'x'}}
  else
    int x; // expected-error{{redefinition of 'x'}}
}
