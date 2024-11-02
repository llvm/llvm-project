// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection,core.builtin \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify %s

void clang_analyzer_eval(int);

void exit(int);

#define UINT_MIN (0U)
#define UINT_MAX (~UINT_MIN)
#define UINT_MID (UINT_MAX / 2 + 1)
#define INT_MAX (UINT_MAX & (UINT_MAX >> 1))
#define INT_MIN (UINT_MAX & ~(UINT_MAX >> 1))

extern void abort() __attribute__((__noreturn__));
#define assert(expr) ((expr) ? (void)(0) : abort())

void assert_in_range(int x) {
  assert(x <= ((int)INT_MAX / 4));
  assert(x >= -(((int)INT_MAX) / 4));
}

void assert_in_wide_range(int x) {
  assert(x <= ((int)INT_MAX / 2));
  assert(x >= -(((int)INT_MAX) / 2));
}

void assert_in_range_2(int m, int n) {
  assert_in_range(m);
  assert_in_range(n);
}

void equal(int m, int n) {
  assert_in_range_2(m, n);
  assert(m == n);
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n == m); // expected-warning{{TRUE}}
}

void non_equal(int m, int n) {
  assert_in_range_2(m, n);
  assert(m != n);
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n != m); // expected-warning{{TRUE}}
}

void less_or_equal(int m, int n) {
  assert_in_range_2(m, n);
  assert(m >= n);
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n <= m); // expected-warning{{TRUE}}
}

void less(int m, int n) {
  assert_in_range_2(m, n);
  assert(m > n);
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n < m); // expected-warning{{TRUE}}
}

void greater_or_equal(int m, int n) {
  assert_in_range_2(m, n);
  assert(m <= n);
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n >= m); // expected-warning{{TRUE}}
}

void greater(int m, int n) {
  assert_in_range_2(m, n);
  assert(m < n);
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n > m); // expected-warning{{TRUE}}
}

void negate_positive_range(int m, int n) {
  assert(m - n > 0);
  clang_analyzer_eval(n - m < 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(n - m > INT_MIN); // expected-warning{{TRUE}}
}

_Static_assert(INT_MIN == -INT_MIN, "");
void negate_int_min(int m, int n) {
  assert(m - n == INT_MIN);
  clang_analyzer_eval(n - m == INT_MIN); // expected-warning{{TRUE}}
}

void negate_mixed(int m, int n) {
  assert(m - n > 0 || m - n == INT_MIN);
  clang_analyzer_eval(n - m <= 0); // expected-warning{{TRUE}}
}

void effective_range(int m, int n) {
  assert(m - n >= 0);
  assert(n - m >= 0);
  clang_analyzer_eval(m - n == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(n - m == 0); // expected-warning{{TRUE}}
}

_Static_assert(INT_MIN == -INT_MIN, "");
void effective_range_2(int m, int n) {
  assert(m - n <= 0);
  assert(n - m <= 0);
  clang_analyzer_eval(m - n == 0 || m - n == INT_MIN); // expected-warning{{TRUE}}
}

void negate_unsigned_min(unsigned m, unsigned n) {
  assert(m - n == UINT_MIN);
  clang_analyzer_eval(n - m == UINT_MIN); // expected-warning{{TRUE}}
  clang_analyzer_eval(n - m != UINT_MIN); // expected-warning{{FALSE}}
  clang_analyzer_eval(n - m > UINT_MIN);  // expected-warning{{FALSE}}
  clang_analyzer_eval(n - m < UINT_MIN);  // expected-warning{{FALSE}}
}

_Static_assert(7u - 3u != 3u - 7u, "");
void negate_unsigned_4(unsigned m, unsigned n) {
  assert(m - n == 4u);
  clang_analyzer_eval(n - m == 4u); // expected-warning{{FALSE}}
  clang_analyzer_eval(n - m != 4u); // expected-warning{{TRUE}}
}

_Static_assert(UINT_MID == -UINT_MID, "");
void negate_unsigned_mid(unsigned m, unsigned n) {
  assert(m - n == UINT_MID);
  clang_analyzer_eval(n - m == UINT_MID); // expected-warning{{TRUE}}
  clang_analyzer_eval(n - m != UINT_MID); // expected-warning{{FALSE}}
}

void negate_unsigned_mid2(unsigned m, unsigned n) {
  assert(UINT_MIN < m - n && m - n < UINT_MID);
  clang_analyzer_eval(n - m > UINT_MID); // expected-warning{{TRUE}}
  clang_analyzer_eval(n - m <= UINT_MAX); // expected-warning{{TRUE}}
}

_Static_assert(1u - 2u == UINT_MAX, "");
_Static_assert(2u - 1u == 1, "");
void negate_unsigned_max(unsigned m, unsigned n) {
  assert(m - n == UINT_MAX);
  clang_analyzer_eval(n - m == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(n - m != 1); // expected-warning{{FALSE}}
}

void negate_unsigned_one(unsigned m, unsigned n) {
  assert(m - n == 1);
  clang_analyzer_eval(n - m == UINT_MAX); // expected-warning{{TRUE}}
  clang_analyzer_eval(n - m < UINT_MAX);  // expected-warning{{FALSE}}
}

// The next code is a repro for the bug PR41588
void negated_unsigned_range(unsigned x, unsigned y) {
  clang_analyzer_eval(x - y != 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(y - x != 0); // expected-warning{{UNKNOWN}}
  // expected no assertion on the next line
  clang_analyzer_eval(x - y != 0); // expected-warning{{UNKNOWN}}
}
