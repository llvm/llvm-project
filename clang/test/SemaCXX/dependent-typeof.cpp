// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<bool B>
void f() {
  decltype(B) x = false;
  __typeof__(B) y = false;
  !x;
  !y;
}
