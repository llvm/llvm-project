// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

// Here, we test that svalbuilder simplification does not produce any
// assertion failure.

void crashing(long a, _Bool b) {
  (void)(a & 1 && 0);
  b = a & 1;
  (void)(b << 1); // expected-warning{{core.UndefinedBinaryOperatorResult}}
}
