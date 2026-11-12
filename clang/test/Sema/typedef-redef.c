// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef const int x; // expected-note {{previous definition is here}}
extern x a;
typedef int x;  // expected-error {{typedef redefinition with different types}}
extern x a;

int y; // expected-note 2 {{previous definition is here}}
float y; // expected-error{{redefinition of 'y' with a different type}}
double y; // expected-error{{redefinition of 'y' with a different type}}
