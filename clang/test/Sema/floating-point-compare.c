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

int f4(float x) {
  return x == 0.0; // no-warning {{comparing}}
}

int f5(float x) {
  return x == __builtin_inf(); // no-warning
}

int f7(float x) {
  return x == 3.14159; // expected-warning {{comparing}}
}
