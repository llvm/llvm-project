// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config support-symbolic-integer-casts=true \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -verify

// Test that the SValBuilder is able to look up and use a constraint for an
// operand of a SymbolCast, when the operand is constrained to a const value.

void clang_analyzer_eval(bool);

extern void abort() __attribute__((__noreturn__));
#define assert(expr) ((expr) ? (void)(0) : abort())

void test(int x) {
  // Constrain a SymSymExpr to a constant value.
  assert(x * x == 1);
  // It is expected to be able to get the constraint for the operand of the
  // cast.
  clang_analyzer_eval((char)(x * x) == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval((long)(x * x) == 1); // expected-warning{{TRUE}}
}

void test1(int x, int y) {
  // Even if two lower bytes of `x` equal to zero, it doesn't mean that
  // the entire `x` is zero. We are not able to know the exact value of x.
  // It can be one of  65536 possible values like
  // [0, 65536, -65536, 131072, -131072, ...]. To avoid huge range sets we
  // still assume `x` in the range [INT_MIN, INT_MAX].
  assert((short)x == 0); // Lower two bytes are set to 0.

  static_assert((short)65536 == 0, "");
  static_assert((short)-65536 == 0, "");
  static_assert((short)131072 == 0, "");
  static_assert((short)-131072 == 0, "");
  clang_analyzer_eval(x == 0);       // expected-warning{{UNKNOWN}}

  // These are not truncated to short as zero.
  static_assert((short)1 != 0, "");
  clang_analyzer_eval(x == 1);       // expected-warning{{FALSE}}
  static_assert((short)-1 != 0, "");
  clang_analyzer_eval(x == -1);      // expected-warning{{FALSE}}
  static_assert((short)65537 != 0, "");
  clang_analyzer_eval(x == 65537);   // expected-warning{{FALSE}}
  static_assert((short)-65537 != 0, "");
  clang_analyzer_eval(x == -65537);  // expected-warning{{FALSE}}
  static_assert((short)131073 != 0, "");
  clang_analyzer_eval(x == 131073);  // expected-warning{{FALSE}}
  static_assert((short)-131073 != 0, "");
  clang_analyzer_eval(x == -131073); // expected-warning{{FALSE}}

  // Check for implicit cast.
  short s = y;
  assert(s == 0);
  clang_analyzer_eval(y == 0); // expected-warning{{UNKNOWN}}
}
