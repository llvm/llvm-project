// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

// expected-no-diagnostics

void clang_analyzer_warnIfReached();

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
