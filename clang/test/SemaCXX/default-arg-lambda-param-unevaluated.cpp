// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

int p(int x, int = sizeof(x));
int q(int x, int = sizeof([] { return x; }()));
int r(int x, int = decltype([] { return x; }())());
int s(int x, bool = noexcept([] { return x; }()));
