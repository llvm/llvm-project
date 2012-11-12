// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

struct S {
  S(int, int) {}
};

void f(int, S const&, int) {}

void test1()
{
  S X1{1, 1,};
  S X2 = {1, 1,};

  f(0, {1, 1}, 0);
}
