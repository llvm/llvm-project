// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

void clang_analyzer_warnIfReached(void);
void clang_analyzer_eval(int);
void clang_analyzer_dump(int);

void remainder_infer_positive_range(int x) {
  if (x < 5 || x > 7)
    return;

  int y = x % 16;
  clang_analyzer_eval(y >= 5 && y <= 7); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

void remainder_infer_positive_range_full(int x) {
  if (x < 9 || x > 51)
    return;

  int y = x % 10;
  clang_analyzer_eval(y >= 0 && y <= 9); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

void remainder_infer_negative_range(int x) {
  if (x < -7 || x > -5)
    return;

  int y = x % 16;
  clang_analyzer_eval(y >= -7 && y <= -5); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

void remainder_infer_positive_range_wraparound(int x) {
  if (x < 30 || x > 33)
    return;

  int y = x % 16;
  clang_analyzer_eval((y >= 0 && y <= 1) || (y >= 14 && y <= 15)); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

void remainder_infer_negative_range_wraparound(int x) {
  if (x < -33 || x > -30)
    return;

  int y = x % 16;
  clang_analyzer_eval((y >= -1 && y <= 0) || (y >= -15 && y <= -14)); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

void remainder_infer_range_spans_zero(int x) {
  if (x < -7 || x > 5)
    return;

  int y = x % 10;
  clang_analyzer_eval(y >= -7 && y <= 5); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

void remainder_infer_mod_one(long long x) {
  long long y = x % 1;
  clang_analyzer_eval(y == 0); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

void remainder_infer_mod_minus_one(long long x) {
  long long y = x % -1;
  clang_analyzer_eval(y == 0); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

#define LONG_MAX 0x7fffffffffffffffL
#define LONG_MIN (-LONG_MAX - 1L)

void remainder_infer_range_mod_long_min(long x) {
  if (x < -7 || x > 5)
    return;

  long y = x % LONG_MIN;
  clang_analyzer_eval(y >= -7 && y <= 5); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

#define INT_MAX 0x7fffffff
#define INT_MIN (-INT_MAX - 1)

void remainder_infer_range_mod_int_min(long x) {
  if (x < -7 || x > 5)
    return;

  int y = x % INT_MIN;
  clang_analyzer_eval(y >= -7 && y <= 5); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}
