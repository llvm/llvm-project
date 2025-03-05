// RUN: %clang_cc1 -std=c++20 %s -verify -fsyntax-only
// expected-no-diagnostics
export module a;
module :private;
static void f() {}
void g() {
  f();
}
