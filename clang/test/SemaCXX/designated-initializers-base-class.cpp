// RUN: %clang_cc1 %s -std=c++1z -fsyntax-only -verify -Winitializer-overrides

struct B {
  int x;
};

struct D : B {
  int y;
};

void test() { D d = {1, .y = 2}; } // expected-warning {{C99 extension}} expected-note {{}}
