// RUN: %clang_analyze_cc1 -std=c++11 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=cplusplus.PlacementNew \
// RUN:   -analyzer-output=text -verify \
// RUN:   -triple x86_64-unknown-linux-gnu

// expected-no-diagnostics

#include "Inputs/system-header-simulator-cxx.h"

struct X {
  static void *operator new(std::size_t sz, void *b) {
    return ::operator new(sz, b);
  }
  long l;
};
void f() {
  short buf;
  X *p1 = new (&buf) X;
  (void)p1;
}
