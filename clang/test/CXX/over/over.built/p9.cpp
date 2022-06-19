// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

struct A{};

template <typename T>
void f(int* pi, A* pa, T* pt) {
  (void)+pi;
  (void)+pa;
  (void)+pt;
}
// expected-no-diagnostics

