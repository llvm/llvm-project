// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

int a(int g, int h = [] { return g; }()); // expected-error {{default argument references parameter 'g'}}
int b(int g, int h = [] { return sizeof(g); }());
int c(int h = [](int x) { int y = x; return y; }(1));
int d(int g, int h = [] { return [] { return g; }(); }()); // expected-error {{default argument references parameter 'g'}}
int e(int g, int h = [] { int g = 1; return g; }());
