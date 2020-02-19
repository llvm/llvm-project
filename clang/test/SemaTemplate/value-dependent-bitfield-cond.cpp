// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template <int b>
class A {
  int c : b;

public:
  void f() {
    if (c)
      ;
  }
};
