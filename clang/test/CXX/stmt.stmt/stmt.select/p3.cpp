// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -std=c++1z -Wc++14-compat -verify %s -DCPP17

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

void ifInitStatement() {
  int Var = 0;

  if (int I = 0; true) {}
  if (Var + Var; true) {}
  if (; true) {}
#ifdef CPP17
  // expected-warning@-4 {{if initialization statements are incompatible with C++ standards before C++17}}
  // expected-warning@-4 {{if initialization statements are incompatible with C++ standards before C++17}}
  // expected-warning@-4 {{if initialization statements are incompatible with C++ standards before C++17}}
#else
  // expected-warning@-8 {{'if' initialization statements are a C++17 extension}}
  // expected-warning@-8 {{'if' initialization statements are a C++17 extension}}
  // expected-warning@-8 {{'if' initialization statements are a C++17 extension}}
#endif
}

void switchInitStatement() {
  int Var = 0;

  switch (int I = 0; Var) {}
  switch (Var + Var; Var) {}
  switch (; Var) {}
#ifdef CPP17
  // expected-warning@-4 {{switch initialization statements are incompatible with C++ standards before C++17}}
  // expected-warning@-4 {{switch initialization statements are incompatible with C++ standards before C++17}}
  // expected-warning@-4 {{switch initialization statements are incompatible with C++ standards before C++17}}
#else
  // expected-warning@-8 {{'switch' initialization statements are a C++17 extension}}
  // expected-warning@-8 {{'switch' initialization statements are a C++17 extension}}
  // expected-warning@-8 {{'switch' initialization statements are a C++17 extension}}
#endif
}

// TODO: Better diagnostics for while init statements.
void whileInitStatement() {
  while (int I = 10; I--); // expected-error {{expected ')'}}
  // expected-note@-1 {{to match this '('}}
  // expected-error@-2 {{use of undeclared identifier 'I'}}

  int Var = 10;
  while (Var + Var; Var--) {} // expected-error {{expected ')'}}
  // expected-note@-1 {{to match this '('}}
  // expected-error@-2 {{expected ';' after expression}}
  // expected-error@-3 {{expected expression}}
}

// TODO: This is needed because clang can't seem to diagnose invalid syntax after the
// last loop above. It would be nice to remove this.
void whileInitStatement2() {
  while (; false) {} // expected-error {{expected expression}}
  // expected-error@-1 {{expected ';' after expression}}
  // expected-error@-2 {{expected expression}}
}
