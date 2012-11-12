// RUN: %clang_cc1 %s -fsyntax-only -verify
int f(int a) { return 0; } // expected-note {{previous definition is here}}
int f(int);
int f(int a) { return 0; } // expected-error {{redefinition of 'f'}}

// <rdar://problem/6097326>
int foo(x) {
  return 0;
}
int x = 1;

// <rdar://problem/6880464>
extern inline int g(void) { return 0; } // expected-note{{previous definition}}
int g(void) { return 0; } // expected-error{{redefinition of a 'extern inline' function 'g' is not supported in C99 mode}}
