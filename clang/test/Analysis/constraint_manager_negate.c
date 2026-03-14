// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection,core.builtin \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify %s

void clang_analyzer_eval(int);
void clang_analyzer_dump(int);

void exit(int);

#define UINT_MIN (0U)
#define UINT_MAX (~UINT_MIN)
#define UINT_MID (UINT_MAX / 2 + 1)
#define INT_MAX (UINT_MAX & (UINT_MAX >> 1))
#define INT_MIN (UINT_MAX & ~(UINT_MAX >> 1))

extern void abort() __attribute__((__noreturn__));
#define assert(expr) ((expr) ? (void)(0) : abort())

void negate_positive_range(int a) {
  assert(-a > 0);
  // -a: [1, INT_MAX]
  // a: [INT_MIN + 1, -1]
  clang_analyzer_eval(a < 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(a > INT_MIN); // expected-warning{{TRUE}}
}

void negate_positive_range2(int a) {
  assert(a > 0);
  // a: [1, INT_MAX]
  // -a: [INT_MIN + 1, -1]
  clang_analyzer_eval(-a < 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(-a > INT_MIN); // expected-warning{{TRUE}}
}

// INT_MIN equals 0b100...00.
// Its two's compelement is 0b011...11 + 1 = 0b100...00 (itself).
_Static_assert(INT_MIN == -INT_MIN, "");
void negate_int_min(int a) {
  assert(a == INT_MIN);
  clang_analyzer_eval(-a == INT_MIN); // expected-warning{{TRUE}}
}

void negate_mixed(int a) {
  assert(a > 0 || a == INT_MIN);
  clang_analyzer_eval(-a <= 0); // expected-warning{{TRUE}}
}

void effective_range(int a) {
  assert(a >= 0);
  assert(-a >= 0);
  clang_analyzer_eval(a == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(-a == 0); // expected-warning{{TRUE}}
}

_Static_assert(-INT_MIN == INT_MIN, "");
void effective_range_2(int a) {
  assert(a <= 0);
  assert(-a <= 0);
  clang_analyzer_eval(a == 0 || a == INT_MIN); // expected-warning{{TRUE}}
}

void negate_symexpr(int a, int b) {
  assert(3 <= a * b && a * b <= 5);
  clang_analyzer_eval(-(a * b) >= -5); // expected-warning{{TRUE}}
  clang_analyzer_eval(-(a * b) <= -3); // expected-warning{{TRUE}}
}

void negate_unsigned_min(unsigned a) {
  assert(a == UINT_MIN);
  clang_analyzer_eval(-a == UINT_MIN); // expected-warning{{TRUE}}
  clang_analyzer_eval(-a != UINT_MIN); // expected-warning{{FALSE}}
  clang_analyzer_eval(-a > UINT_MIN);  // expected-warning{{FALSE}}
  clang_analyzer_eval(-a < UINT_MIN);  // expected-warning{{FALSE}}
}

_Static_assert(7u - 3u != 3u - 7u, "");
_Static_assert(-4u == UINT_MAX - 3u, "");
void negate_unsigned_4(unsigned a) {
  assert(a == 4u);
  clang_analyzer_eval(-a == 4u); // expected-warning{{FALSE}}
  clang_analyzer_eval(-a != 4u); // expected-warning{{TRUE}}
  clang_analyzer_eval(-a == UINT_MAX - 3u); // expected-warning{{TRUE}}
}

// UINT_MID equals 0b100...00.
// Its two's compelement is 0b011...11 + 1 = 0b100...00 (itself).
_Static_assert(UINT_MID == -UINT_MID, "");
void negate_unsigned_mid(unsigned a) {
  assert(a == UINT_MID);
  clang_analyzer_eval(-a == UINT_MID); // expected-warning{{TRUE}}
  clang_analyzer_eval(-a != UINT_MID); // expected-warning{{FALSE}}
}

void negate_unsigned_mid2(unsigned a) {
  assert(UINT_MIN < a && a < UINT_MID);
  // a:  [UINT_MIN+1, UINT_MID-1]
  // -a: [UINT_MID+1, UINT_MAX]
  clang_analyzer_eval(-a > UINT_MID); // expected-warning{{TRUE}}
  clang_analyzer_eval(-a <= UINT_MAX); // expected-warning{{TRUE}}
}

_Static_assert(1u - 2u == UINT_MAX, "");
_Static_assert(2u - 1u == 1, "");
void negate_unsigned_max(unsigned a) {
  assert(a == UINT_MAX);
  clang_analyzer_eval(-a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(-a != 1); // expected-warning{{FALSE}}
}
void negate_unsigned_one(unsigned a) {
  assert(a == 1);
  clang_analyzer_eval(-a == UINT_MAX); // expected-warning{{TRUE}}
  clang_analyzer_eval(-a < UINT_MAX);  // expected-warning{{FALSE}}
}
