// RUN: %clang_cc1 -fsyntax-only -std=c++11 -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -pedantic -verify %s

#define for 0    // expected-warning {{keyword is hidden by macro definition}}
#define final 1  // expected-warning {{keyword is hidden by macro definition}}
#define override // expected-warning {{keyword is hidden by macro definition}}

int x;
