// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-constraints=unsupported-z3 -verify %s
// REQUIRES: z3
// expected-no-diagnostics

void atomic_bool(_Bool input) {
  _Atomic(_Bool) value = input;
  if (value) {
  }
}

typedef _Bool B1;
typedef _Bool B2;

void atomic_bool_typedef(B1 input) {
  _Atomic(B2) value = input;
  if (value) {
  }
}
