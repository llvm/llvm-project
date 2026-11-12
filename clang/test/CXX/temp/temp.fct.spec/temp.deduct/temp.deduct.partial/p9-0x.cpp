// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T> int &f0(T&);
template<typename T> float &f0(T&&);

// Core issue 1164
void test_f0(int i) {
  int &ir0 = f0(i);
  float &fr0 = f0(5);
}
