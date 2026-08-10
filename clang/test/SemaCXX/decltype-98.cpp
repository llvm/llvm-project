// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify %s
// expected-no-diagnostics
extern int x;
__decltype(1) x = 3;
