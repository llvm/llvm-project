// RUN: %clang_cc1 -fsyntax-only -verify -std=c2y %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify=pedantic -std=c23 -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify=pedantic -x c++ -pedantic %s
// expected-no-diagnostics

void f() {
  x: while (1) break x; // pedantic-warning {{labeled 'break' is a C2y extension}}
}
