// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s

template <typename T>
struct A {
  // Used to crash when field was named after class.
  int A = 0;
};
A<int> a;
