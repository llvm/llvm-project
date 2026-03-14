// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s
// expected-no-diagnostics

// Do not crash on initialization to complex numbers.
void init_complex() {
  _Complex float valid1 = { 0.0f, 0.0f };
}
