// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -Wc++14-binary-literal
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -Wc++14-extensions

int x = 0b11;
// expected-warning@-1{{binary integer literals are a C++14 extension}}
