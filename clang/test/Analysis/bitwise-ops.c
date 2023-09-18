// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-disable-checker=core.BitwiseShift -triple x86_64-apple-darwin13 -Wno-shift-count-overflow -verify %s

// NOTE: This test file disables the checker core.BitwiseShift (which would
// report all undefined behavior connected to bitwise shifts) to verify the
// behavior of core.UndefinedBinaryOperatorResult (which resports cases when
// the constant folding in BasicValueFactory produces an "undefined" result
// from a shift or any other binary operator).

void clang_analyzer_eval(int);
#define CHECK(expr) if (!(expr)) return; clang_analyzer_eval(expr)

void testPersistentConstraints(int x, int y) {
  // Basic check
  CHECK(x); // expected-warning{{TRUE}}
  CHECK(x & 1); // expected-warning{{TRUE}}
  
  CHECK(1 - x); // expected-warning{{TRUE}}
  CHECK(x & y); // expected-warning{{TRUE}}
}

int testConstantShifts_PR18073(int which) {
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

int testUnrepresentableLeftShift(int a) {
  if (a == 8)
    return a << 30; // expected-warning{{The result of the left shift is undefined due to shifting '8' by '30', which is unrepresentable in the unsigned version of the return type 'int'}}
  return 0;
}
