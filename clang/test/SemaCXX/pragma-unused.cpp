// RUN: %clang_cc1 -fsyntax-only -Wunused-parameter -Wunused -verify %s
// expected-no-diagnostics

struct S {
  void m(int x, int y) {
    int z;
    #pragma unused(x,y,z)
  }
};
