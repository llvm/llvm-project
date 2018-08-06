// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection,core.builtin -analyzer-config aggressive-relational-comparison-simplification=true -verify %s

void clang_analyzer_eval(int);

void exit(int);

#define UINT_MAX (~0U)
#define INT_MAX (UINT_MAX & (UINT_MAX >> 1))
#define INT_MIN (UINT_MAX & ~(UINT_MAX >> 1))

extern void __assert_fail (__const char *__assertion, __const char *__file,
    unsigned int __line, __const char *__function)
     __attribute__ ((__noreturn__));
#define assert(expr) \
  ((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))

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
  if (m != n)
    return;
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n == m); // expected-warning{{TRUE}}
}

void non_equal(int m, int n) {
  assert_in_range_2(m, n);
  if (m == n)
    return;
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n != m); // expected-warning{{TRUE}}
}

void less_or_equal(int m, int n) {
  assert_in_range_2(m, n);
  if (m < n)
    return;
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n <= m); // expected-warning{{TRUE}}
}

void less(int m, int n) {
  assert_in_range_2(m, n);
  if (m <= n)
    return;
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n < m); // expected-warning{{TRUE}}
}

void greater_or_equal(int m, int n) {
  assert_in_range_2(m, n);
  if (m > n)
    return;
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n >= m); // expected-warning{{TRUE}}
}

void greater(int m, int n) {
  assert_in_range_2(m, n);
  if (m >= n)
    return;
  assert_in_wide_range(m - n);
  clang_analyzer_eval(n > m); // expected-warning{{TRUE}}
}

void negate_positive_range(int m, int n) {
  if (m - n <= 0)
    return;
  clang_analyzer_eval(n - m < 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(n - m > INT_MIN); // expected-warning{{TRUE}}
  clang_analyzer_eval(n - m == INT_MIN); // expected-warning{{FALSE}}
}

void negate_int_min(int m, int n) {
  if (m - n != INT_MIN)
    return;
  clang_analyzer_eval(n - m == INT_MIN); // expected-warning{{TRUE}}
}

void negate_mixed(int m, int n) {
  if (m -n > INT_MIN && m - n <= 0)
    return;
  clang_analyzer_eval(n - m <= 0); // expected-warning{{TRUE}}
}
