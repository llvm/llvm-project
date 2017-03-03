// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

#define UINT_MAX (~0U)
#define INT_MAX (UINT_MAX & (UINT_MAX >> 1))
#define INT_MIN (-INT_MAX - 1)


// Each of these adjusted ranges has an adjustment small enough to split the
// solution range across an overflow boundary (Min for <, Max for >).
// This corresponds to one set of branches in RangeConstraintManager.
void smallAdjustmentGT (unsigned a) {
  if (a+2 > 1)
    clang_analyzer_eval(a < UINT_MAX-1); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a == UINT_MAX-1 || a == UINT_MAX); // expected-warning{{TRUE}}
}

void smallAdjustmentGE (unsigned a) {
  if (a+2 >= 1)
    clang_analyzer_eval(a < UINT_MAX-1 || a == UINT_MAX); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a == UINT_MAX-1); // expected-warning{{TRUE}}
}

void smallAdjustmentLT (unsigned a) {
  if (a+1 < 2)
    clang_analyzer_eval(a == 0 || a == UINT_MAX); // expected-warning{{TRUE}}
}

void smallAdjustmentLE (unsigned a) {
  if (a+1 <= 2)
    clang_analyzer_eval(a == 0 || a == 1 || a == UINT_MAX); // expected-warning{{TRUE}}
}


// Each of these adjusted ranges has an adjustment large enough to push the
// comparison value over an overflow boundary (Min for <, Max for >).
// This corresponds to one set of branches in RangeConstraintManager.
void largeAdjustmentGT (unsigned a) {
  if (a-2 > UINT_MAX-1)
    clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a != 1); // expected-warning{{TRUE}}
}

void largeAdjustmentGE (unsigned a) {
  if (a-2 >= UINT_MAX-1)
    clang_analyzer_eval(a == 1 || a == 0); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a > 1); // expected-warning{{TRUE}}
}

void largeAdjustmentLT (unsigned a) {
  if (a+2 < 1)
    clang_analyzer_eval(a == UINT_MAX-1); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a != UINT_MAX-1); // expected-warning{{TRUE}}
}

void largeAdjustmentLE (unsigned a) {
  if (a+2 <= 1)
    clang_analyzer_eval(a == UINT_MAX-1 || a == UINT_MAX); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a < UINT_MAX-1); // expected-warning{{TRUE}}
}


// Test the nine cases in RangeConstraintManager's pinning logic.
// For out-of-range tautologies, it may be the negation that actually
// triggers the case in question.
void mixedComparisons1(signed char a) {
  // Case 1: The range is entirely below the symbol's range.
  int min = INT_MIN;

  clang_analyzer_eval((a - 2) >= (min + 5LL)); // expected-warning{{TRUE}}

  clang_analyzer_eval(a == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == 0x7F); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == -0x80); // expected-warning{{UNKNOWN}}
}

void mixedComparisons2(signed char a) {
  // Case 2: Only the lower end of the range is outside.
  clang_analyzer_eval((a - 5) < (-0x81LL)); // expected-warning{{UNKNOWN}}

  if ((a - 5) < (-0x81LL)) {
    clang_analyzer_eval(a == 0); // expected-warning{{FALSE}}
    clang_analyzer_eval(a == 0x7F); // expected-warning{{FALSE}}
    clang_analyzer_eval(a == -0x80); // expected-warning{{UNKNOWN}}
  }
}

void mixedComparisons3(signed char a) {
  // Case 3: The entire symbol range is covered.
  clang_analyzer_eval((a - 0x200) < -0x100LL); // expected-warning{{TRUE}}

  clang_analyzer_eval(a == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == 0x7F); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == -0x80); // expected-warning{{UNKNOWN}}
}

void mixedComparisons4(signed char a) {
  // Case 4: The range wraps around, but the lower wrap is out-of-range.
  clang_analyzer_eval((a - 5) > 0LL); // expected-warning{{UNKNOWN}}

  if ((a - 5) > 0LL) {
    clang_analyzer_eval(a == 0); // expected-warning{{FALSE}}
    clang_analyzer_eval(a == 0x7F); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(a == -0x80); // expected-warning{{FALSE}}
  }
}

void mixedComparisons5(signed char a) {
  // Case 5: The range is inside and may or may not wrap.
  clang_analyzer_eval((a + 5) == 0LL); // expected-warning{{UNKNOWN}}

  if ((a + 5) == 0LL) {
    clang_analyzer_eval(a == 0); // expected-warning{{FALSE}}
    clang_analyzer_eval(a == 0x7F); // expected-warning{{FALSE}}
    clang_analyzer_eval(a == -0x80); // expected-warning{{FALSE}}
  } else {
    clang_analyzer_eval(a == 0); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(a == 0x7F); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(a == -0x80); // expected-warning{{UNKNOWN}}
  }
}

void mixedComparisons6(signed char a) {
  // Case 6: Only the upper end of the range is outside.
  clang_analyzer_eval((a + 5) > 0x81LL); // expected-warning{{UNKNOWN}}

  if ((a + 5) > 0x81LL) {
    clang_analyzer_eval(a == 0); // expected-warning{{FALSE}}
    clang_analyzer_eval(a == 0x7F); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(a == -0x80); // expected-warning{{FALSE}}
  }
}

void mixedComparisons7(signed char a) {
  // Case 7: The range wraps around but is entirely outside the symbol's range.
  int min = INT_MIN;

  clang_analyzer_eval((a + 2) >= (min + 5LL)); // expected-warning{{TRUE}}

  clang_analyzer_eval(a == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == 0x7F); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == -0x80); // expected-warning{{UNKNOWN}}
}

void mixedComparisons8(signed char a) {
  // Case 8: The range wraps, but the upper wrap is out of range.
  clang_analyzer_eval((a + 5) < 0LL); // expected-warning{{UNKNOWN}}

  if ((a + 5) < 0LL) {
    clang_analyzer_eval(a == 0); // expected-warning{{FALSE}}
    clang_analyzer_eval(a == 0x7F); // expected-warning{{FALSE}}
    clang_analyzer_eval(a == -0x80); // expected-warning{{UNKNOWN}}
  }
}

void mixedComparisons9(signed char a) {
  // Case 9: The range is entirely above the symbol's range.
  int max = INT_MAX;

  clang_analyzer_eval((a + 2) <= (max - 5LL)); // expected-warning{{TRUE}}

  clang_analyzer_eval(a == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == 0x7F); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == -0x80); // expected-warning{{UNKNOWN}}
}


void mixedSignedness1(int a) {
  unsigned max = UINT_MAX;
  clang_analyzer_eval(a < max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) < max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) < max); // expected-warning{{UNKNOWN}}
}

void mixedSignedness2(int a) {
  unsigned max = UINT_MAX;
  clang_analyzer_eval(a <= max); // expected-warning{{TRUE}}
  clang_analyzer_eval((a + 2) <= max); // expected-warning{{TRUE}}
  clang_analyzer_eval((a + 2U) <= max); // expected-warning{{TRUE}}
}

void mixedSignedness3(unsigned a) {
  int max = INT_MAX;
  clang_analyzer_eval(a < max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) < max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) < max); // expected-warning{{UNKNOWN}}
}

void mixedSignedness4(unsigned a) {
  int max = INT_MAX;
  clang_analyzer_eval(a <= max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) <= max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) <= max); // expected-warning{{UNKNOWN}}
}

void mixedSignedness5(unsigned a) {
  int min = INT_MIN;
  clang_analyzer_eval(a < min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) < min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) < min); // expected-warning{{UNKNOWN}}
}

void mixedSignedness6(unsigned a) {
  int min = INT_MIN;
  clang_analyzer_eval(a <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) <= min); // expected-warning{{UNKNOWN}}
}

void mixedSignedness7(unsigned a) {
  unsigned min = 0;
  clang_analyzer_eval(a < min); // expected-warning{{FALSE}}
  clang_analyzer_eval((a + 2) < min); // expected-warning{{FALSE}}
  clang_analyzer_eval((a + 2U) < min); // expected-warning{{FALSE}}
}

void mixedSignedness8(unsigned a) {
  unsigned min = 0;
  clang_analyzer_eval(a <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) <= min); // expected-warning{{UNKNOWN}}
}

void mixedSignedness9(unsigned a) {
  int min = 0;
  clang_analyzer_eval(a < min); // expected-warning{{FALSE}}
  clang_analyzer_eval((a + 2) < min); // expected-warning{{FALSE}}
  clang_analyzer_eval((a + 2U) < min); // expected-warning{{FALSE}}
}

void mixedSignedness10(unsigned a) {
  int min = 0;
  clang_analyzer_eval(a <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) <= min); // expected-warning{{UNKNOWN}}
}

void mixedSignedness11(int a) {
  int min = 0;
  clang_analyzer_eval(a < min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) < min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) < min); // expected-warning{{FALSE}}
}

void mixedSignedness12(int a) {
  int min = 0;
  clang_analyzer_eval(a <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) <= min); // expected-warning{{UNKNOWN}}
}

void mixedSignedness13(int a) {
  unsigned max = INT_MAX;
  clang_analyzer_eval(a < max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) < max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) < max); // expected-warning{{UNKNOWN}}
}

void mixedSignedness14(int a) {
  unsigned max = INT_MAX;
  clang_analyzer_eval(a <= max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) <= max); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) <= max); // expected-warning{{UNKNOWN}}
}

void mixedSignedness15(int a) {
  unsigned min = INT_MIN;
  clang_analyzer_eval(a < min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) < min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) < min); // expected-warning{{UNKNOWN}}
}

void mixedSignedness16(int a) {
  unsigned min = INT_MIN;
  clang_analyzer_eval(a <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2) <= min); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a + 2U) <= min); // expected-warning{{UNKNOWN}}
}

void mixedSignedness17(int a) {
  unsigned max = INT_MAX;
  if (a < max)
    return;

  clang_analyzer_eval(a < 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(a == INT_MAX); // expected-warning{{UNKNOWN}}
}

void mixedSignedness18(int a) {
  if (a >= 0)
    return;

  clang_analyzer_eval(a < 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(a == (unsigned)INT_MIN); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == UINT_MAX); // expected-warning{{UNKNOWN}}
}
