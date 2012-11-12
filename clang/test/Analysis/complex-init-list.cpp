// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-output=text -verify %s
// expected-no-diagnostics

// Do not crash on initialization to complex numbers.
void init_complex() {
  _Complex float valid1 = { 0.0f, 0.0f };
}
