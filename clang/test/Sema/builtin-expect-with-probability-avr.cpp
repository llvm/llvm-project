// RUN: %clang_cc1 -triple avr -fsyntax-only -verify %s

void test(int x, double p) { // expected-note {{declared here}}
  bool dummy = false;
  dummy = __builtin_expect_with_probability(x > 0, 1, 0.9);
  dummy = __builtin_expect_with_probability(x > 0, 1, 1.1); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, -1); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, p); // expected-error {{probability argument to __builtin_expect_with_probability must be constant floating-point expression}} expected-note {{function parameter 'p' with unknown value}}
  dummy = __builtin_expect_with_probability(x > 0, 1, "aa"); // expected-error {{cannot initialize a parameter of type 'double' with an lvalue of type 'const char[3]'}}
  dummy = __builtin_expect_with_probability(x > 0, 1, __builtin_nan("")); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, __builtin_inf()); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, -0.0);
  dummy = __builtin_expect_with_probability(x > 0, 1, 1.0 + __DBL_EPSILON__); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, -__DBL_DENORM_MIN__); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
}
