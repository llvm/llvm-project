// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyzer-output=text \
// RUN:   -verify %s

int track_mul_lhs_0(int x, int y) {
  int p0 = x < 0;   // expected-note {{Assuming 'x' is >= 0}} \
                    // expected-note {{'p0' initialized to 0}}
  int div = p0 * y; // expected-note {{'div' initialized to 0}}
  return 1 / div;   // expected-note {{Division by zero}} \
                    // expected-warning {{Division by zero}}
}
