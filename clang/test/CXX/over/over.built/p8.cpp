// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T>
void f(void(*pf)(), T(*ptf)(T)) {
  (void)*pf;
  (void)*ptf;
}
// expected-no-diagnostics

