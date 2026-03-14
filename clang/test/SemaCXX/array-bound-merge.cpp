// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR5515

extern int a[];
int a[10];
extern int b[10];
int b[];
extern int c[1];
int c[] = {1,2}; // expected-error {{excess elements in array initializer}}

int d[1][]; // expected-error {{array has incomplete element type 'int[]'}}

extern const int e[2]; // expected-note {{previous declaration is here}}
int e[] = { 1 }; // expected-error {{redefinition of 'e' with a different type: 'int[]' vs 'const int[2]'}}
