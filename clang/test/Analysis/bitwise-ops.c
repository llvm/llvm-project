// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -triple x86_64-apple-darwin13 -Wno-shift-count-overflow -verify %s

void clang_analyzer_eval(int);
#define CHECK(expr) if (!(expr)) return; clang_analyzer_eval(expr)

void testPersistentConstraints(int x, int y) {
  // Sanity check
  CHECK(x); // expected-warning{{TRUE}}
  CHECK(x & 1); // expected-warning{{TRUE}}
  
  // False positives due to SValBuilder giving up on certain kinds of exprs.
  CHECK(1 - x); // expected-warning{{UNKNOWN}}
  CHECK(x & y); // expected-warning{{UNKNOWN}}
}

int testConstantShifts_PR18073(int which) {
  // FIXME: We should have a checker that actually specifically checks bitwise
  // shifts against the width of the LHS's /static/ type, rather than just
  // having BasicValueFactory return "undefined" when dealing with two constant
  // operands.
  switch (which) {
  case 1:
    return 0ULL << 63; // no-warning
  case 2:
    return 0ULL << 64; // expected-warning{{The result of the left shift is undefined due to shifting by '64', which is greater or equal to the width of type 'unsigned long long'}}
  case 3:
    return 0ULL << 65; // expected-warning{{The result of the left shift is undefined due to shifting by '65', which is greater or equal to the width of type 'unsigned long long'}}

  default:
    return 0;
  }
}

int testOverflowShift(int a) {
  if (a == 323) {
    return 1 << a; // expected-warning{{The result of the left shift is undefined due to shifting by '323', which is greater or equal to the width of type 'int'}}
  }
  return 0;
}

int testNegativeShift(int a) {
  if (a == -5) {
    return 1 << a; // expected-warning{{The result of the left shift is undefined because the right operand is negative}}
  }
  return 0;
}

int testNegativeLeftShift(int a) {
  if (a == -3) {
    return a << 1; // expected-warning{{The result of the left shift is undefined because the left operand is negative}}
  }
  return 0;
}
