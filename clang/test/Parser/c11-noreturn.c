// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -std=c99 -pedantic -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-EXT %s

_Noreturn int f();
int _Noreturn f(); // expected-note {{previous}}
int f _Noreturn(); // expected-error {{expected ';'}} expected-error 2{{}}
int f() _Noreturn; // expected-error {{'_Noreturn' keyword must precede function declarator}}

_Noreturn char c1; // expected-error {{'_Noreturn' can only appear on functions}}
char _Noreturn c2; // expected-error {{'_Noreturn' can only appear on functions}}

typedef _Noreturn int g(); // expected-error {{'_Noreturn' can only appear on functions}}

_Noreturn int; // expected-error {{'_Noreturn' can only appear on functions}} expected-warning {{does not declare anything}}
_Noreturn struct S; // expected-error {{'_Noreturn' can only appear on functions}}
_Noreturn enum E { e }; // expected-error {{'_Noreturn' can only appear on functions}}

// CHECK-EXT: '_Noreturn' is a C11 extension
