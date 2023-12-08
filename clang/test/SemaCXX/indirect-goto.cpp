// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace test1 {
  // Make sure this doesn't crash.
  struct A { ~A(); };
  void a() { goto *(A(), &&L); L: return; }
}
