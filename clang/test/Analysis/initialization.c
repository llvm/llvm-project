// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

void initbug() {
  const union { float a; } u = {};
  (void)u.a; // no-crash
}
