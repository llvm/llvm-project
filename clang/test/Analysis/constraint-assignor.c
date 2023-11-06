// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

void clang_analyzer_warnIfReached(void);
void clang_analyzer_eval(int);

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

void gh_62215(int x, int y, int z) {
  if (x != y) return; // x == y
  if (z <= x) return; // z > x
  if (z >= y) return; // z < y
  clang_analyzer_warnIfReached(); // no-warning: This should be dead code.
  (void)(x + y + z); // keep the constraints alive.
}

void gh_62215_contradicting_right_equivalent(int x, int y, int z) {
  if (x == y && z > x) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}

    // `z < y` should mean the same thing as `z < x`, which would contradict with `z > x`
    if (z < y) {
      clang_analyzer_warnIfReached(); // no-warning: dead code
    }
  }
  (void)(x + y + z); // keep the constraints alive.
}

void gh_62215_contradicting_left_equivalent(int x, int y, int z) {
  if (x == y && z > x) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}

    // `y > z` should mean the same thing as `x > z`, which would contradict with `z > x`
    if (y > z) {
      clang_analyzer_warnIfReached(); // no-warning
    }
  }
  (void)(x + y + z); // keep the constraints alive.
}

void gh_62215_left_and_right(int x, int y, int z, int w) {
  if (x != y) return; // x == y
  if (z != w) return; // z == w
  if (z <= x) return; // z > x
  if (w >= y) return; // w < y
  // FIXME: We fail to recognize that `w` and `y` are equivalent with `x` and `z`
  // respectively and recognize the contradiction.
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}} should be dead code
  (void)(x + y + z + w);
}

void gh_62215_contradicting_nested_right_equivalent(int x, int y, int z) {
  if (y > 1 && y < 10) { // y: [2,9]
    if (x == y && z > x) {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}

      // `z < (y - 1)` should mean the same thing as `z < (x - 1)`, which
      // should contradict with `z > x` (assuming x,y: [2,9])
      if (z < (y - 1)) {
        // FIXME: This should be dead code.
        clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}} Z3 crosscheck eliminate this btw
      }
    }
  }
  (void)(x + y + z); // keep the constraints alive.
}
