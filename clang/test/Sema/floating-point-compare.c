// RUN: %clang_cc1 -fsyntax-only -Wfloat-equal -verify %s

int f1(float x, float y) {
  return x == y; // expected-warning {{comparing floating point with ==}}
} 

int f2(float x, float y) {
  return x != y; // expected-warning {{comparing floating point with ==}}
}

int f3(float x) {
  return x == x; // no-warning
}

// 0.0 can be represented exactly, so don't warn.
int f4(float x) {
  return x == 0.0; // no-warning {{comparing}}
}

int f5(float x) {
  return x == __builtin_inf(); // no-warning
}

// The literal is a double that can't be represented losslessly as a float.
int tautological_FP_compare(float x) {
  return x == 3.14159; // expected-warning {{floating-point comparison is always false}}
}

int tautological_FP_compare_inverse(float x) {
  return x != 3.14159; // expected-warning {{floating-point comparison is always true}}
}

// The literal is a double that can be represented losslessly as a long double,
// but this might not be the comparison what was intended.
int not_tautological_FP_compare(long double f) {
  return f == 0.1; // expected-warning {{comparing floating point with ==}}
}

int tautological_FP_compare_commute(float f) {
  return 0.3 == f; // expected-warning {{floating-point comparison is always false}}
}

int tautological_FP_compare_commute_inverse(float f) {
  return 0.3 != f; // expected-warning {{floating-point comparison is always true}}
}

int tautological_FP_compare_explicit_upcast(float f) {
  return 0.3 == (double) f; // expected-warning {{floating-point comparison is always false}}
}

int tautological_FP_compare_explicit_downcast(double d) {
  return 0.3 == (float) d; // expected-warning {{floating-point comparison is always false}}
}

int tautological_FP_compare_ignore_parens(float f) {
  return f == (0.3); // expected-warning {{floating-point comparison is always false}}
}

#define CST 0.3

int tautological_FP_compare_macro(float f) {
  return f == CST; // expected-warning {{floating-point comparison is always false}}
}
