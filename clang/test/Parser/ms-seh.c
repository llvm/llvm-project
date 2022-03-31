// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-extensions

void f(void) {
  int a;

  __try a; // expected-error {{expected '{'}} expected-warning {{expression result unused}}

  __try {
  }
} // expected-error {{expected '__except' or '__finally' block}}

void g(void) {
  int a;

  __try {
  } __except(1) a; // expected-error {{expected '{'}} expected-warning {{expression result unused}}
}

void h(void) {
  int a;

  __try {
  } __finally a; // expected-error {{expected '{'}} expected-warning {{expression result unused}}
}
