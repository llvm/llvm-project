// RUN: %clang_cc1 -fsyntax-only -verify -std=c2y %s
// RUN: %clang_cc1 -fsyntax-only -verify -fnamed-loops -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify -fnamed-loops -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify=disabled -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=disabled -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify=disabled -std=c23 -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify=disabled -x c++ -pedantic %s
// expected-no-diagnostics

void f() {
  x: while (1) break x; // disabled-error {{labeled 'break' is only supported in C2y}}
}
