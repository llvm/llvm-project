// RUN: %clang_cc1 -fsyntax-only -verify -Wformat -std=c++11 %s
// expected-no-diagnostics
extern "C" {
extern int scanf(const char *restrict, ...);
extern int printf(const char *restrict, ...);
}

void f(char *c) {
  printf("%p", c);
}
