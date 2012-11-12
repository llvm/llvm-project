// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

extern int a1[];
int a1[1];

extern int a2[]; // expected-note {{previous definition is here}}
float a2[1]; // expected-error {{redefinition of 'a2'}}

extern int a3[][2];
int a3[1][2];

extern int a4[][2]; // expected-note {{previous definition is here}}
int a4[2]; // expected-error {{redefinition of 'a4'}}

extern int a5[1][2][3]; // expected-note {{previous definition is here}}
int a5[3][2][1]; // expected-error {{redefinition of 'a5'}}
