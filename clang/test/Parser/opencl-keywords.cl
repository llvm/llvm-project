// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

void f(half *h) {
  bool b;
  int wchar_t;
  int constexpr;
}
