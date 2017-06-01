// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

class B {
public:
  [[clang::disable_tail_calls]] virtual int foo1() { return 1; }
  [[clang::disable_tail_calls]] int foo2() { return 2; }
};
