// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

void clang_analyzer_eval(int);

void div_infer_positive_divider(int x) {
  if (x < 1 || x > 5)
    return;

  int i = x / 2;
  clang_analyzer_eval(i >= 0 && i <= 2); // expected-warning{{TRUE}}
}

void div_infer_negative_divider_positive_range(int x) {
  if (x < 1 || x > 2)
    return;

  int i = x / -2;
  clang_analyzer_eval(i >= -1 && i <= 0); // expected-warning{{TRUE}}
}

void div_infer_negative_divider_negative_range(int x) {
  if (x < -2 || x > 0)
    return;

  int i = x / -2;
  clang_analyzer_eval(i >= 0 && i <= 1); // expected-warning{{TRUE}}
}

void div_infer_positive_divider_positive_range(int x) {
  if (x < 0 || x > 5)
    return;

  int i = x / 6;
  clang_analyzer_eval(i == 0); // expected-warning{{TRUE}}
}

#define LONG_MAX 0x7fffffffffffffffLL
#define LONG_MIN (-LONG_MAX - 1LL)

void div_infer_overflow_long(long long x) {
  if (x > LONG_MIN + 1)
    return;

  // x in [LONG_MIN, LONG_MIN + 1]
  clang_analyzer_eval(x >= LONG_MIN && x <= LONG_MIN + 1); // expected-warning{{TRUE}}
  long long i = x / -1;
  clang_analyzer_eval(i == LONG_MIN || i == LONG_MAX); // expected-warning{{TRUE}}
}

#define INT_MAX 0x7fffffff
#define INT_MIN (-INT_MAX - 1)

void div_infer_overflow_int(int x) {
  if (x > INT_MIN + 1)
    return;

  // x in [INT_MIN, INT_MIN + 1]
  clang_analyzer_eval(x >= INT_MIN && x <= INT_MIN + 1); // expected-warning{{TRUE}}
  int i = x / -1;
  clang_analyzer_eval(i == INT_MIN || i == INT_MAX); // expected-warning{{TRUE}}
}
