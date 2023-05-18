// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace pr7801 {
  extern void* x[];
  void* dummy[] = { &x };
}
