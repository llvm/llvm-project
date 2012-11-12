// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

// There should be no warnings unless otherwise indicated.

void testComparisons (int a) {
  // Sema can already catch the simple comparison a==a,
  // since that's usually a logic error (and not path-dependent).
  int b = a;
  clang_analyzer_eval(b == a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b >= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b <= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b != a); // expected-warning{{FALSE}}
  clang_analyzer_eval(b > a); // expected-warning{{FALSE}}
  clang_analyzer_eval(b < a); // expected-warning{{FALSE}}
}

void testSelfOperations (int a) {
  clang_analyzer_eval((a|a) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a&a) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a^a) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((a-a) == 0); // expected-warning{{TRUE}}
}

void testIdempotent (int a) {
  clang_analyzer_eval((a*1) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a/1) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a+0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a-0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a<<0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a>>0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a^0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a&(~0)) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a|0) == a); // expected-warning{{TRUE}}
}

void testReductionToConstant (int a) {
  clang_analyzer_eval((a*0) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((a&0) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((a|(~0)) == (~0)); // expected-warning{{TRUE}}
}

void testSymmetricIntSymOperations (int a) {
  clang_analyzer_eval((2+a) == (a+2)); // expected-warning{{TRUE}}
  clang_analyzer_eval((2*a) == (a*2)); // expected-warning{{TRUE}}
  clang_analyzer_eval((2&a) == (a&2)); // expected-warning{{TRUE}}
  clang_analyzer_eval((2^a) == (a^2)); // expected-warning{{TRUE}}
  clang_analyzer_eval((2|a) == (a|2)); // expected-warning{{TRUE}}
}

void testAsymmetricIntSymOperations (int a) {
  clang_analyzer_eval(((~0) >> a) == (~0)); // expected-warning{{TRUE}}
  clang_analyzer_eval((0 >> a) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((0 << a) == 0); // expected-warning{{TRUE}}

  // Unsigned right shift shifts in zeroes.
  clang_analyzer_eval(((~0U) >> a) != (~0U)); // expected-warning{{UNKNOWN}}
}

void testLocations (char *a) {
  char *b = a;
  clang_analyzer_eval(b == a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b >= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b <= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b != a); // expected-warning{{FALSE}}
  clang_analyzer_eval(b > a); // expected-warning{{FALSE}}
  clang_analyzer_eval(b < a); // expected-warning{{FALSE}}
}

void testMixedTypeComparisons (char a, unsigned long b) {
  if (a != 0) return;
  if (b != 0x100) return;

  clang_analyzer_eval(a <= b); // expected-warning{{TRUE}}
  clang_analyzer_eval(b >= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(a != b); // expected-warning{{TRUE}}
}
