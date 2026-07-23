// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-constraints=unsupported-z3 -verify %s
// REQUIRES: z3

void clang_analyzer_eval(int);

void unsigned_bitint_8_wrap_equivalence(unsigned _BitInt(8) x) {
  clang_analyzer_eval( // expected-warning{{TRUE}}
      (x + (unsigned _BitInt(8))1 == 0) ==
      (x == (unsigned _BitInt(8))255));
}

void unsigned_bitint_35_wrap_equivalence(unsigned _BitInt(35) x) {
  clang_analyzer_eval( // expected-warning{{TRUE}}
      (x + 1 == 0) ==
      (x == (unsigned _BitInt(35))0x7ffffffffULL));
}

void unsigned_bitint_63_wrap_equivalence(unsigned _BitInt(63) x) {
  // Addition wraps modulo 2^63, so these two conditions are equivalent.
  clang_analyzer_eval( // expected-warning{{TRUE}}
      (x + 1 == 0) ==
      (x == (unsigned _BitInt(63))0x7fffffffffffffffULL));
}

void unsigned_bitint_256_wrap_equivalence(unsigned _BitInt(256) x) {
  clang_analyzer_eval( // expected-warning{{TRUE}}
      (x + 1 == 0) ==
      (x == ~(unsigned _BitInt(256))0));
}

void unsigned_bitint_widening_cast(unsigned _BitInt(35) x) {
  clang_analyzer_eval( // expected-warning{{TRUE}}
      (unsigned _BitInt(63))x <=
      (unsigned _BitInt(63))0x7ffffffffULL);
}
