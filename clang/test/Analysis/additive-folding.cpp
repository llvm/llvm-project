// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -Wno-tautological-compare -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);

#define UINT_MAX (~0U)
#define INT_MAX (UINT_MAX & (UINT_MAX >> 1))
#define INT_MIN (-INT_MAX - 1)

//---------------
//  Plus/minus
//---------------

void separateExpressions (int a) {
  int b = a + 1;
  --b;

  clang_analyzer_eval(a != 0 && b == 0); // expected-warning{{FALSE}}
}

void oneLongExpression (int a) {
  // Expression canonicalization should still allow this to work, even though
  // the first term is on the left.
  int b = 15 + a + 15 - 10 - 20;

  clang_analyzer_eval(a != 0 && b == 0); // expected-warning{{FALSE}}
}

void mixedTypes (int a) {
  // Different additive types should not cause crashes when constant-folding.
  // This is part of PR7406.
  int b = a + 1LL;
  clang_analyzer_eval(a != 0 && (b-1) == 0); // not crash, expected-warning{{FALSE}}

  int c = a + 1U;
  clang_analyzer_eval(a != 0 && (c-1) == 0); // not crash, expected-warning{{FALSE}}
}

//---------------
//  Comparisons
//---------------

// Equality and inequality only
void eq_ne (unsigned a) {
  if (a == UINT_MAX) {
    clang_analyzer_eval(a+1 == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(a-1 == UINT_MAX-1); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(a+1 != 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(a-1 != UINT_MAX-1); // expected-warning{{TRUE}}
  }
}

// Mixed typed inequalities (part of PR7406)
// These should not crash.
void mixed_eq_ne (int a) {
  if (a == 1) {
    clang_analyzer_eval(a+1U == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a-1U == 0); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(a+1U != 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a-1U != 0); // expected-warning{{TRUE}}
  }
}


// Simple order comparisons with no adjustment
void baselineGT (unsigned a) {
  if (a > 0)
    clang_analyzer_eval(a != 0); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a == 0); // expected-warning{{TRUE}}
}

void baselineGE (unsigned a) {
  if (a >= UINT_MAX)
    clang_analyzer_eval(a == UINT_MAX); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a != UINT_MAX); // expected-warning{{TRUE}}
}

void baselineLT (unsigned a) {
  if (a < UINT_MAX)
    clang_analyzer_eval(a != UINT_MAX); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a == UINT_MAX); // expected-warning{{TRUE}}
}

void baselineLE (unsigned a) {
  if (a <= 0)
    clang_analyzer_eval(a == 0); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(a != 0); // expected-warning{{TRUE}}
}


// Adjustment gives each of these an extra solution!
void adjustedGT (unsigned a) {
  clang_analyzer_eval(a-1 > UINT_MAX-1); // expected-warning{{UNKNOWN}}
}

void adjustedGE (unsigned a) {
  clang_analyzer_eval(a-1 > UINT_MAX-1); // expected-warning{{UNKNOWN}}

  if (a-1 >= UINT_MAX-1)
    clang_analyzer_eval(a == UINT_MAX); // expected-warning{{UNKNOWN}}
}

void adjustedLT (unsigned a) {
  clang_analyzer_eval(a+1 < 1); // expected-warning{{UNKNOWN}}
}

void adjustedLE (unsigned a) {
  clang_analyzer_eval(a+1 <= 1); // expected-warning{{UNKNOWN}}

  if (a+1 <= 1)
    clang_analyzer_eval(a == 0); // expected-warning{{UNKNOWN}}
}


// Tautologies
// The negative forms are exercised as well
// because clang_analyzer_eval tests both possibilities.
void tautologies(unsigned a) {
  clang_analyzer_eval(a <= UINT_MAX); // expected-warning{{TRUE}}
  clang_analyzer_eval(a >= 0); // expected-warning{{TRUE}}
}


// Tautologies from outside the range of the symbol
void tautologiesOutside(unsigned char a) {
  clang_analyzer_eval(a <= 0x100); // expected-warning{{TRUE}}
  clang_analyzer_eval(a < 0x100); // expected-warning{{TRUE}}

  clang_analyzer_eval(a != 0x100); // expected-warning{{TRUE}}
  clang_analyzer_eval(a != -1); // expected-warning{{TRUE}}

  clang_analyzer_eval(a > -1); // expected-warning{{TRUE}}
  clang_analyzer_eval(a >= -1); // expected-warning{{TRUE}}
}


// Wraparound with mixed types. Note that the analyzer assumes
// -fwrapv semantics.
void mixedWraparoundSanityCheck(int a) {
  int max = INT_MAX;
  int min = INT_MIN;

  int b = a + 1;
  clang_analyzer_eval(a == max && b != min); // expected-warning{{FALSE}}
}

void mixedWraparoundLE_GT(int a) {
  int max = INT_MAX;
  int min = INT_MIN;

  clang_analyzer_eval((a + 2) <= (max + 1LL)); // expected-warning{{TRUE}}
  clang_analyzer_eval((a - 2) > (min - 1LL)); // expected-warning{{TRUE}}
  clang_analyzer_eval((a + 2LL) <= max); // expected-warning{{UNKNOWN}}
}

void mixedWraparoundGE_LT(int a) {
  int max = INT_MAX;
  int min = INT_MIN;

  clang_analyzer_eval((a + 2) < (max + 1LL)); // expected-warning{{TRUE}}
  clang_analyzer_eval((a - 2) >= (min - 1LL)); // expected-warning{{TRUE}}
  clang_analyzer_eval((a - 2LL) >= min); // expected-warning{{UNKNOWN}}
}

void mixedWraparoundEQ_NE(int a) {
  int max = INT_MAX;

  clang_analyzer_eval((a + 2) != (max + 1LL)); // expected-warning{{TRUE}}
  clang_analyzer_eval((a + 2LL) == (max + 1LL)); // expected-warning{{UNKNOWN}}
}


// Mixed-signedness comparisons.
void mixedSignedness(int a, unsigned b) {
  int sMin = INT_MIN;
  unsigned uMin = INT_MIN;

  clang_analyzer_eval(a == sMin && a != uMin); // expected-warning{{FALSE}}
  clang_analyzer_eval(b == uMin && b != sMin); // expected-warning{{FALSE}}
}

void mixedSignedness2(int a) {
  if (a != -1)
    return;
  clang_analyzer_eval(a == UINT_MAX); // expected-warning{{TRUE}}
}

void mixedSignedness3(unsigned a) {
  if (a != UINT_MAX)
    return;
  clang_analyzer_eval(a == -1); // expected-warning{{TRUE}}
}


void multiplicativeSanityTest(int x) {
  // At one point we were ignoring the *4 completely -- the constraint manager
  // would see x < 8 and then declare the assertion to be known false.
  if (x*4 < 8)
    return;

  clang_analyzer_eval(x == 3); // expected-warning{{UNKNOWN}}
}

void additiveSymSymFolding(int x, int y) {
  // We should simplify 'x - 1' to '0' and handle the comparison,
  // despite both sides being complicated symbols.
  int z = x - 1;
  if (x == 1)
    if (y >= 0)
      clang_analyzer_eval(z <= y); // expected-warning{{TRUE}}
}
