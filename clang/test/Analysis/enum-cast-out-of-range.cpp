// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,alpha.cplusplus.EnumCastOutOfRange \
// RUN:   -std=c++11 -verify %s

enum unscoped_unspecified_t {
  unscoped_unspecified_0 = -4,
  unscoped_unspecified_1,
  unscoped_unspecified_2 = 1,
  unscoped_unspecified_3,
  unscoped_unspecified_4 = 4
};

enum unscoped_specified_t : int {
  unscoped_specified_0 = -4,
  unscoped_specified_1,
  unscoped_specified_2 = 1,
  unscoped_specified_3,
  unscoped_specified_4 = 4
};

enum class scoped_unspecified_t {
  scoped_unspecified_0 = -4,
  scoped_unspecified_1,
  scoped_unspecified_2 = 1,
  scoped_unspecified_3,
  scoped_unspecified_4 = 4
};

enum class scoped_specified_t : int {
  scoped_specified_0 = -4,
  scoped_specified_1,
  scoped_specified_2 = 1,
  scoped_specified_3,
  scoped_specified_4 = 4
};

struct S {
  unscoped_unspecified_t E : 5;
};

void unscopedUnspecified() {
  unscoped_unspecified_t InvalidBeforeRangeBegin = static_cast<unscoped_unspecified_t>(-5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t ValidNegativeValue1 = static_cast<unscoped_unspecified_t>(-4); // OK.
  unscoped_unspecified_t ValidNegativeValue2 = static_cast<unscoped_unspecified_t>(-3); // OK.
  unscoped_unspecified_t InvalidInsideRange1 = static_cast<unscoped_unspecified_t>(-2); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t InvalidInsideRange2 = static_cast<unscoped_unspecified_t>(-1); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t InvalidInsideRange3 = static_cast<unscoped_unspecified_t>(0); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t ValidPositiveValue1 = static_cast<unscoped_unspecified_t>(1); // OK.
  unscoped_unspecified_t ValidPositiveValue2 = static_cast<unscoped_unspecified_t>(2); // OK.
  unscoped_unspecified_t InvalidInsideRange4 = static_cast<unscoped_unspecified_t>(3); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t ValidPositiveValue3 = static_cast<unscoped_unspecified_t>(4); // OK.
  unscoped_unspecified_t InvalidAfterRangeEnd = static_cast<unscoped_unspecified_t>(5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void unscopedSpecified() {
  unscoped_specified_t InvalidBeforeRangeBegin = static_cast<unscoped_specified_t>(-5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t ValidNegativeValue1 = static_cast<unscoped_specified_t>(-4); // OK.
  unscoped_specified_t ValidNegativeValue2 = static_cast<unscoped_specified_t>(-3); // OK.
  unscoped_specified_t InvalidInsideRange1 = static_cast<unscoped_specified_t>(-2); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t InvalidInsideRange2 = static_cast<unscoped_specified_t>(-1); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t InvalidInsideRange3 = static_cast<unscoped_specified_t>(0); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t ValidPositiveValue1 = static_cast<unscoped_specified_t>(1); // OK.
  unscoped_specified_t ValidPositiveValue2 = static_cast<unscoped_specified_t>(2); // OK.
  unscoped_specified_t InvalidInsideRange4 = static_cast<unscoped_specified_t>(3); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t ValidPositiveValue3 = static_cast<unscoped_specified_t>(4); // OK.
  unscoped_specified_t InvalidAfterRangeEnd = static_cast<unscoped_specified_t>(5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void scopedUnspecified() {
  scoped_unspecified_t InvalidBeforeRangeBegin = static_cast<scoped_unspecified_t>(-5); // expected-warning{{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t ValidNegativeValue1 = static_cast<scoped_unspecified_t>(-4); // OK.
  scoped_unspecified_t ValidNegativeValue2 = static_cast<scoped_unspecified_t>(-3); // OK.
  scoped_unspecified_t InvalidInsideRange1 = static_cast<scoped_unspecified_t>(-2); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t InvalidInsideRange2 = static_cast<scoped_unspecified_t>(-1); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t InvalidInsideRange3 = static_cast<scoped_unspecified_t>(0); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t ValidPositiveValue1 = static_cast<scoped_unspecified_t>(1); // OK.
  scoped_unspecified_t ValidPositiveValue2 = static_cast<scoped_unspecified_t>(2); // OK.
  scoped_unspecified_t InvalidInsideRange4 = static_cast<scoped_unspecified_t>(3); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t ValidPositiveValue3 = static_cast<scoped_unspecified_t>(4); // OK.
  scoped_unspecified_t InvalidAfterRangeEnd = static_cast<scoped_unspecified_t>(5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void scopedSpecified() {
  scoped_specified_t InvalidBeforeRangeBegin = static_cast<scoped_specified_t>(-5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t ValidNegativeValue1 = static_cast<scoped_specified_t>(-4); // OK.
  scoped_specified_t ValidNegativeValue2 = static_cast<scoped_specified_t>(-3); // OK.
  scoped_specified_t InvalidInsideRange1 = static_cast<scoped_specified_t>(-2); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t InvalidInsideRange2 = static_cast<scoped_specified_t>(-1); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t InvalidInsideRange3 = static_cast<scoped_specified_t>(0); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t ValidPositiveValue1 = static_cast<scoped_specified_t>(1); // OK.
  scoped_specified_t ValidPositiveValue2 = static_cast<scoped_specified_t>(2); // OK.
  scoped_specified_t InvalidInsideRange4 = static_cast<scoped_specified_t>(3); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t ValidPositiveValue3 = static_cast<scoped_specified_t>(4); // OK.
  scoped_specified_t InvalidAfterRangeEnd = static_cast<scoped_specified_t>(5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void unscopedUnspecifiedCStyle() {
  unscoped_unspecified_t InvalidBeforeRangeBegin = (unscoped_unspecified_t)(-5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t ValidNegativeValue1 = (unscoped_unspecified_t)(-4); // OK.
  unscoped_unspecified_t ValidNegativeValue2 = (unscoped_unspecified_t)(-3); // OK.
  unscoped_unspecified_t InvalidInsideRange1 = (unscoped_unspecified_t)(-2); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t InvalidInsideRange2 = (unscoped_unspecified_t)(-1); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t InvalidInsideRange3 = (unscoped_unspecified_t)(0); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t ValidPositiveValue1 = (unscoped_unspecified_t)(1); // OK.
  unscoped_unspecified_t ValidPositiveValue2 = (unscoped_unspecified_t)(2); // OK.
  unscoped_unspecified_t InvalidInsideRange4 = (unscoped_unspecified_t)(3); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_unspecified_t ValidPositiveValue3 = (unscoped_unspecified_t)(4); // OK.
  unscoped_unspecified_t InvalidAfterRangeEnd = (unscoped_unspecified_t)(5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void unscopedSpecifiedCStyle() {
  unscoped_specified_t InvalidBeforeRangeBegin = (unscoped_specified_t)(-5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t ValidNegativeValue1 = (unscoped_specified_t)(-4); // OK.
  unscoped_specified_t ValidNegativeValue2 = (unscoped_specified_t)(-3); // OK.
  unscoped_specified_t InvalidInsideRange1 = (unscoped_specified_t)(-2); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t InvalidInsideRange2 = (unscoped_specified_t)(-1); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t InvalidInsideRange3 = (unscoped_specified_t)(0); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t ValidPositiveValue1 = (unscoped_specified_t)(1); // OK.
  unscoped_specified_t ValidPositiveValue2 = (unscoped_specified_t)(2); // OK.
  unscoped_specified_t InvalidInsideRange4 = (unscoped_specified_t)(3); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  unscoped_specified_t ValidPositiveValue3 = (unscoped_specified_t)(4); // OK.
  unscoped_specified_t InvalidAfterRangeEnd = (unscoped_specified_t)(5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void scopedUnspecifiedCStyle() {
  scoped_unspecified_t InvalidBeforeRangeBegin = (scoped_unspecified_t)(-5); // expected-warning{{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t ValidNegativeValue1 = (scoped_unspecified_t)(-4); // OK.
  scoped_unspecified_t ValidNegativeValue2 = (scoped_unspecified_t)(-3); // OK.
  scoped_unspecified_t InvalidInsideRange1 = (scoped_unspecified_t)(-2); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t InvalidInsideRange2 = (scoped_unspecified_t)(-1); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t InvalidInsideRange3 = (scoped_unspecified_t)(0); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t ValidPositiveValue1 = (scoped_unspecified_t)(1); // OK.
  scoped_unspecified_t ValidPositiveValue2 = (scoped_unspecified_t)(2); // OK.
  scoped_unspecified_t InvalidInsideRange4 = (scoped_unspecified_t)(3); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_unspecified_t ValidPositiveValue3 = (scoped_unspecified_t)(4); // OK.
  scoped_unspecified_t InvalidAfterRangeEnd = (scoped_unspecified_t)(5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void scopedSpecifiedCStyle() {
  scoped_specified_t InvalidBeforeRangeBegin = (scoped_specified_t)(-5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t ValidNegativeValue1 = (scoped_specified_t)(-4); // OK.
  scoped_specified_t ValidNegativeValue2 = (scoped_specified_t)(-3); // OK.
  scoped_specified_t InvalidInsideRange1 = (scoped_specified_t)(-2); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t InvalidInsideRange2 = (scoped_specified_t)(-1); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t InvalidInsideRange3 = (scoped_specified_t)(0); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t ValidPositiveValue1 = (scoped_specified_t)(1); // OK.
  scoped_specified_t ValidPositiveValue2 = (scoped_specified_t)(2); // OK.
  scoped_specified_t InvalidInsideRange4 = (scoped_specified_t)(3); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
  scoped_specified_t ValidPositiveValue3 = (scoped_specified_t)(4); // OK.
  scoped_specified_t InvalidAfterRangeEnd = (scoped_specified_t)(5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

unscoped_unspecified_t unused;
void unusedExpr() {
  // following line is not something that EnumCastOutOfRangeChecker should evaluate.  checker should either ignore this line
  // or process it without producing any warnings.  However, compilation will (and should) still generate a warning having
  // nothing to do with this checker.
  unused; // expected-warning {{expression result unused}}
}

void rangeConstrained1(int input) {
  if (input > -5 && input < 5)
    auto value = static_cast<scoped_specified_t>(input); // OK. Being conservative, this is a possibly good value.
}

void rangeConstrained2(int input) {
  if (input < -5)
    auto value = static_cast<scoped_specified_t>(input); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void rangeConstrained3(int input) {
  if (input >= -2 && input <= -1)
    auto value = static_cast<scoped_specified_t>(input); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void rangeConstrained4(int input) {
  if (input >= -2 && input <= 1)
    auto value = static_cast<scoped_specified_t>(input); // OK. Possibly 1.
}

void rangeConstrained5(int input) {
  if (input >= 1 && input <= 2)
    auto value = static_cast<scoped_specified_t>(input); // OK. Strict inner matching.
}

void rangeConstrained6(int input) {
  if (input >= 2 && input <= 4)
    auto value = static_cast<scoped_specified_t>(input); // OK. The value is possibly 2 or 4, dont warn.
}

void rangeConstrained7(int input) {
  if (input >= 3 && input <= 3)
    auto value = static_cast<scoped_specified_t>(input); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}

void enumBitFieldAssignment() {
  S s;
  s.E = static_cast<unscoped_unspecified_t>(4); // OK.
  s.E = static_cast<unscoped_unspecified_t>(5); // expected-warning {{The value provided to the cast expression is not in the valid range of values for the enum}}
}
