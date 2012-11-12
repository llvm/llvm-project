// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -pedantic -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-EXT %s

_Alignas(4) char c1;
unsigned _Alignas(long) char c2;
char _Alignas(16) c3;

char c4 _Alignas(32); // expected-error {{expected ';' after top level declarator}}

char _Alignas(_Alignof(int)) c5;

// CHECK-EXT: _Alignas is a C11-specific feature
// CHECK-EXT: _Alignof is a C11-specific feature
