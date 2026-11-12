// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

void foo(int operator+); // expected-error{{'operator+' cannot be the name of a parameter}}
