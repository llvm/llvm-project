// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++ [basic.def.odr]p1:
//   No translation unit shall contain more than one definition of any
//   variable, [...].

// Bad: in C++, these are both definitions. None of that C99 tentative stuff.
int i; // expected-note {{previous}}
int i; // expected-error {{redefinition}}

// OK: decl + def
extern int j;
int j;

// OK: def + decl
int k;
extern int k;

// Bad. The important thing here is that we don't emit the diagnostic twice.
int l = 1; // expected-note {{previous}}
int l = 2; // expected-error {{redefinition}}
