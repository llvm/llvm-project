// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=true \
// RUN:   -verify

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=true \
// RUN:   -analyzer-config support-symbolic-integer-casts=true \
// RUN:   -verify

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=true \
// RUN:   -analyzer-config crosscheck-with-z3=true \
// RUN:   -verify

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=true \
// RUN:   -analyzer-config crosscheck-with-z3=true \
// RUN:   -analyzer-config support-symbolic-integer-casts=true \
// RUN:   -verify

// REQUIRES: z3

void k(long L) {
  int g = L;
  int h = g + 1;
  int j;
  j += -h < 0; // should not crash
  // expected-warning@-1{{garbage}}
}
