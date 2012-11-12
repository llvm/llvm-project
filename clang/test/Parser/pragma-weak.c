// RUN: %clang_cc1 -fsyntax-only -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

int x;
/* expected-warning {{expected identifier in '#pragma weak'}}*/ #pragma weak
#pragma weak x

extern int z;
/* expected-warning {{expected identifier in '#pragma weak'}}*/ #pragma weak z = =
/* expected-warning {{expected identifier in '#pragma weak'}}*/ #pragma weak z =
/* expected-warning {{weak identifier 'y' never declared}} */ #pragma weak z = y

extern int a;
/* expected-warning {{extra tokens at end of '#pragma weak'}}*/ #pragma weak a b
/* expected-warning {{extra tokens at end of '#pragma weak'}}*/ #pragma weak a = x c
