// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s

void f(int x);                   // expected-note {{previous declaration is here}}
void f([[indeterminate]] int x); // expected-error {{'[[indeterminate]]' attribute on parameter 'x' must appear on the first declaration of the function}}

void g([[indeterminate]] int x);
void g([[indeterminate]] int x) {}

void h([[indeterminate]] int x);
void h(int x) {}
