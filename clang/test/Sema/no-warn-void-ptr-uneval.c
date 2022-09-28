// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify %s

// expected-no-diagnostics
void foo(void *vp) {
  sizeof(*vp);
  sizeof(*(vp));
  void inner(typeof(*vp));
}
