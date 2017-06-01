// RUN: %clang_cc1 %s -std=c++1z -fsyntax-only -verify -Winitializer-overrides
// expected-no-diagnostics

struct B {
  int x;
};

struct D : B {
  int y;
};

void test() { D d = {1, .y = 2}; }
