// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,alpha.cplusplus.EnumCastOutOfRange \
// RUN:   -verify %s

enum En_t {
  En_0 = -4,
  En_1,
  En_2 = 1,
  En_3,
  En_4 = 4
};

void unscopedUnspecifiedCStyle() {
  enum En_t Below = (enum En_t)(-5);    // expected-warning {{not in the valid range}}
  enum En_t NegVal1 = (enum En_t)(-4);  // OK.
  enum En_t NegVal2 = (enum En_t)(-3);  // OK.
  enum En_t InRange1 = (enum En_t)(-2); // expected-warning {{not in the valid range}}
  enum En_t InRange2 = (enum En_t)(-1); // expected-warning {{not in the valid range}}
  enum En_t InRange3 = (enum En_t)(0);  // expected-warning {{not in the valid range}}
  enum En_t PosVal1 = (enum En_t)(1);   // OK.
  enum En_t PosVal2 = (enum En_t)(2);   // OK.
  enum En_t InRange4 = (enum En_t)(3);  // expected-warning {{not in the valid range}}
  enum En_t PosVal3 = (enum En_t)(4);   // OK.
  enum En_t Above = (enum En_t)(5);     // expected-warning {{not in the valid range}}
}

enum En_t unused;
void unusedExpr() {
  // Following line is not something that EnumCastOutOfRangeChecker should
  // evaluate.  Checker should either ignore this line or process it without
  // producing any warnings.  However, compilation will (and should) still
  // generate a warning having nothing to do with this checker.
  unused; // expected-warning {{expression result unused}}
}
