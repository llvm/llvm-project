// RUN: %clang_cc1 -std=c++98 -verify %s
// RUN: %clang_cc1 -std=c++1z -verify %s

// expected-no-diagnostics

struct A { A(); A(int); };
void f() {
  const A a;
  true ? a : 0;
}
