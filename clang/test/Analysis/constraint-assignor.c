// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

void clang_analyzer_warnIfReached(void);
void clang_analyzer_eval(int);
void clang_analyzer_dump(int);

void rem_constant_rhs_ne_zero(int x, int y) {
  if (x % 3 == 0) // x % 3 != 0 -> x != 0
    return;
  if (x * y != 0) // x * y == 0
    return;
  if (y != 1)     // y == 1     -> x == 0
    return;
  clang_analyzer_warnIfReached(); // no-warning
  (void)x; // keep the constraints alive.
}

void rem_symbolic_rhs_ne_zero(int x, int y, int z) {
  if (x % z == 0) // x % z != 0 -> x != 0
    return;
  if (x * y != 0) // x * y == 0
    return;
  if (y != 1)     // y == 1     -> x == 0
    return;
  clang_analyzer_warnIfReached(); // no-warning
  (void)x; // keep the constraints alive.
}

void rem_symbolic_rhs_ne_zero_nested(int w, int x, int y, int z) {
  if (w % x % z == 0) // w % x % z != 0 -> w % x != 0
    return;
  if (w % x * y != 0) // w % x * y == 0
    return;
  if (y != 1)         // y == 1         -> w % x == 0
    return;
  clang_analyzer_warnIfReached(); // no-warning
  (void)(w * x); // keep the constraints alive.
}

void rem_constant_rhs_ne_zero_early_contradiction(int x, int y) {
  if ((x + y) != 0)     // (x + y) == 0
    return;
  if ((x + y) % 3 == 0) // (x + y) % 3 != 0 -> (x + y) != 0 -> contradiction
    return;
  clang_analyzer_warnIfReached(); // no-warning
  (void)x; // keep the constraints alive.
}

void rem_symbolic_rhs_ne_zero_early_contradiction(int x, int y, int z) {
  if ((x + y) != 0)     // (x + y) == 0
    return;
  if ((x + y) % z == 0) // (x + y) % z != 0 -> (x + y) != 0 -> contradiction
    return;
  clang_analyzer_warnIfReached(); // no-warning
  (void)x; // keep the constraints alive.
}

void internal_unsigned_signed_mismatch(unsigned a) {
  int d = a;
  // Implicit casts are not handled, thus the analyzer models `d % 2` as
  // `(reg_$0<unsigned int a>) % 2`
  // However, this should not result in internal signedness mismatch error when
  // we assign new constraints below.
  if (d % 2 != 0)
    return;
}

void remainder_with_adjustment(int x) {
  if ((x + 1) % 3 == 0) // (x + 1) % 3 != 0 -> x + 1 != 0 -> x != -1
    return;
  clang_analyzer_eval(x + 1 != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(x != -1);    // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

void remainder_with_adjustment_of_composit_lhs(int x, int y) {
  if ((x + y + 1) % 3 == 0) // (x + 1) % 3 != 0 -> x + 1 != 0 -> x != -1
    return;
  clang_analyzer_eval(x + y + 1 != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(x + y != -1);    // expected-warning{{TRUE}}
  (void)(x * y); // keep the constraints alive.
}

void remainder_infeasible_positive_range(int x) {
  if (x <= 2 || x >= 5)
    return;
  if (x % 5 != 0)
    return;
  clang_analyzer_warnIfReached(); // no-warning
  (void)x; // keep the constraints alive.
}

void remainder_infeasible_negative_range(int x) {
  if (x <= -14 || x >= -1)
    return;
  if (x % 15 != 0)
    return;
  clang_analyzer_warnIfReached(); // no-warning
  (void)x; // keep the constraints alive.
}

void remainder_within_modulo_positive_range_unsigned_1(unsigned x) {
  if (x <= 2 || x > 6)
    return;
  if (x % 5 != 0)
    return;
  clang_analyzer_dump(x); // expected-warning{{5 S32b}}
  (void)x; // keep the constraints alive.
}

void remainder_within_modulo_positive_range_unsigned_2(unsigned char x) {
  if (x < 252 || x > 254)
    return;
  if (x % 5 != 0)
    return;
  clang_analyzer_dump(x); // no-warning
  (void)x; // keep the constraints alive.
}

void remainder_within_modulo_positive_range_unsigned_3(unsigned x) {
  if (x < 4294967289 || x > 4294967294)
    return;
  if (x % 10 != 0)
    return;
  clang_analyzer_eval(x == 4294967290); // expected-warning{{TRUE}}
  (void)x; // keep the constraints alive.
}

void remainder_within_modulo_positive_range(int x) {
  if (x <= 2 || x > 6)
    return;
  if (x % 5 != 0)
    return;
  clang_analyzer_dump(x); // expected-warning{{5 S32b}}
  (void)x; // keep the constraints alive.
}

void remainder_within_modulo_range_spans_zero(int x) {
  if (x <= -2 || x > 2)
    return;
  if (x % 5 != 0)
    return;
  clang_analyzer_dump(x); // expected-warning{{0 S32b}}
  (void)x; // keep the constraints alive.
}

void remainder_within_modulo_negative_range(int x) {
  if (x <= -7 || x > -2)
    return;
  if (x % 5 != 0)
    return;
  clang_analyzer_dump(x); // expected-warning{{-5 S32b}}
  (void)x; // keep the constraints alive.
}

void remainder_within_modulo_range_neg_mod(int x) {
  if (x <= 2 || x > 6)
    return;
  if (x % -5 != 0)
    return;
  clang_analyzer_dump(x); // expected-warning{{5 S32b}}
  (void)x; // keep the constraints alive.
}

#define LONG_MAX 0x7fffffffffffffffLL
#define LONG_MIN (-LONG_MAX - 1LL)

void remainder_within_modulo_distance_overflow(long long x) {
  if (x < LONG_MIN + 1 || x > LONG_MAX - 1)
    return;

  if (x % 10 != 0)
    return;
  clang_analyzer_dump(x); // expected-warning{{reg_$0<long long x>}}
  (void)x; // keep the constraints alive.
}

#define CHAR_MAX 0x7f
#define CHAR_MIN (-CHAR_MAX - 1)

void remainder_within_modulo_not_overflow(char x) {
  if (x < CHAR_MIN + 1 || x > CHAR_MAX - 1)
    return;

  if (x % (CHAR_MAX * 2) != 0)
    return;
  clang_analyzer_dump(x); // expected-warning{{0 S32b}}
  (void)x; // keep the constraints alive.
}
