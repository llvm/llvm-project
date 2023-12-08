// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete,cplusplus.NewDeleteLeaks,unix.Malloc -std=c++11 -fblocks -verify %s
// expected-no-diagnostics

namespace std {
  typedef __typeof__(sizeof(int)) size_t;
}

struct X {};

void *operator new(std::size_t, X, ...);
void *operator new[](std::size_t, X, ...);

void testGlobalCustomVariadicNew() {
  X x;

  void *p1 = operator new(0, x); // no warn

  void *p2 = operator new[](0, x); // no warn

  int *p3 = new (x) int; // no warn

  int *p4 = new (x) int[0]; // no warn
}
