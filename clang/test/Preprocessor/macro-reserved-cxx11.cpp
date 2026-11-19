// RUN: %clang_cc1 -fsyntax-only -std=c++11 -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -pedantic -verify %s

#define for 0    // expected-error {{keyword is hidden by macro definition}}

// final and override are contextual keyword, only diagnose if it's a real keyword.
#define final 1
#define override

int x;
