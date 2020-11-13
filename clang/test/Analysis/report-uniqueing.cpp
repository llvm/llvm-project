// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=security

void bzero(void *, unsigned long);

template <typename T> void foo(T l) {
  // The warning comes from multiple instances and with
  // different declarations that have same source location.
  // One instance should be shown.
  bzero(l, 1); // expected-warning{{The bzero() function is obsoleted}}
}

void p(int *p, unsigned *q) {
  foo(p);
  foo(q);
}
