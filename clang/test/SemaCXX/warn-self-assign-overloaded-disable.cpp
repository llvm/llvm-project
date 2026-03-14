// RUN: %clang_cc1 -fsyntax-only -Wall -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign-overloaded -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wall -Wno-self-assign-overloaded -DSILENCE -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign -Wno-self-assign-overloaded -DSILENCE -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign-overloaded -Wno-self-assign-overloaded -DSILENCE -verify %s

struct S {};

void f() {
  S a;
#ifndef SILENCE
  a = a; // expected-warning{{explicitly assigning}}
#else
  // expected-no-diagnostics
  a = a;
#endif
}
