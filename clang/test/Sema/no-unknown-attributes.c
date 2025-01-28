// RUN: %clang_cc1 -fsyntax-only -Wno-unknown-attributes -std=c23 -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unknown-attributes -std=c++2b -verify %s

// expected-no-diagnostics

[[unknown_ns::a]][[gnu::b]]
int f(void) {
  return 0;
}
