// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,optin.core.EnumCastOutOfRange \
// RUN:   -std=c++11 -verify %s

// expected-note@+1 + {{enum declared here}}
enum unscoped_unspecified_t {
  unscoped_unspecified_0 = -4,
  unscoped_unspecified_1,
  unscoped_unspecified_2 = 1,
  unscoped_unspecified_3,
  unscoped_unspecified_4 = 4
};

// expected-note@+1 + {{enum declared here}}
enum unscoped_specified_t : int {
  unscoped_specified_0 = -4,
  unscoped_specified_1,
  unscoped_specified_2 = 1,
  unscoped_specified_3,
  unscoped_specified_4 = 4
};

// expected-note@+1 + {{enum declared here}}
enum class scoped_unspecified_t {
  scoped_unspecified_0 = -4,
  scoped_unspecified_1,
  scoped_unspecified_2 = 1,
  scoped_unspecified_3,
  scoped_unspecified_4 = 4
};

// expected-note@+1 + {{enum declared here}}
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
  unscoped_unspecified_t InvalidBeforeRangeBegin = static_cast<unscoped_unspecified_t>(-5); // expected-warning {{The value '-5' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t ValidNegativeValue1 = static_cast<unscoped_unspecified_t>(-4); // OK.
  unscoped_unspecified_t ValidNegativeValue2 = static_cast<unscoped_unspecified_t>(-3); // OK.
  unscoped_unspecified_t InvalidInsideRange1 = static_cast<unscoped_unspecified_t>(-2); // expected-warning {{The value '-2' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t InvalidInsideRange2 = static_cast<unscoped_unspecified_t>(-1); // expected-warning {{The value '-1' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t InvalidInsideRange3 = static_cast<unscoped_unspecified_t>(0); // expected-warning {{The value '0' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t ValidPositiveValue1 = static_cast<unscoped_unspecified_t>(1); // OK.
  unscoped_unspecified_t ValidPositiveValue2 = static_cast<unscoped_unspecified_t>(2); // OK.
  unscoped_unspecified_t InvalidInsideRange4 = static_cast<unscoped_unspecified_t>(3); // expected-warning {{The value '3' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t ValidPositiveValue3 = static_cast<unscoped_unspecified_t>(4); // OK.
  unscoped_unspecified_t InvalidAfterRangeEnd = static_cast<unscoped_unspecified_t>(5); // expected-warning {{The value '5' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
}

void unscopedSpecified() {
  unscoped_specified_t InvalidBeforeRangeBegin = static_cast<unscoped_specified_t>(-5); // expected-warning {{The value '-5' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t ValidNegativeValue1 = static_cast<unscoped_specified_t>(-4); // OK.
  unscoped_specified_t ValidNegativeValue2 = static_cast<unscoped_specified_t>(-3); // OK.
  unscoped_specified_t InvalidInsideRange1 = static_cast<unscoped_specified_t>(-2); // expected-warning {{The value '-2' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t InvalidInsideRange2 = static_cast<unscoped_specified_t>(-1); // expected-warning {{The value '-1' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t InvalidInsideRange3 = static_cast<unscoped_specified_t>(0); // expected-warning {{The value '0' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t ValidPositiveValue1 = static_cast<unscoped_specified_t>(1); // OK.
  unscoped_specified_t ValidPositiveValue2 = static_cast<unscoped_specified_t>(2); // OK.
  unscoped_specified_t InvalidInsideRange4 = static_cast<unscoped_specified_t>(3); // expected-warning {{The value '3' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t ValidPositiveValue3 = static_cast<unscoped_specified_t>(4); // OK.
  unscoped_specified_t InvalidAfterRangeEnd = static_cast<unscoped_specified_t>(5); // expected-warning {{The value '5' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
}

void scopedUnspecified() {
  scoped_unspecified_t InvalidBeforeRangeBegin = static_cast<scoped_unspecified_t>(-5); // expected-warning{{The value '-5' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t ValidNegativeValue1 = static_cast<scoped_unspecified_t>(-4); // OK.
  scoped_unspecified_t ValidNegativeValue2 = static_cast<scoped_unspecified_t>(-3); // OK.
  scoped_unspecified_t InvalidInsideRange1 = static_cast<scoped_unspecified_t>(-2); // expected-warning {{The value '-2' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t InvalidInsideRange2 = static_cast<scoped_unspecified_t>(-1); // expected-warning {{The value '-1' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t InvalidInsideRange3 = static_cast<scoped_unspecified_t>(0); // expected-warning {{The value '0' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t ValidPositiveValue1 = static_cast<scoped_unspecified_t>(1); // OK.
  scoped_unspecified_t ValidPositiveValue2 = static_cast<scoped_unspecified_t>(2); // OK.
  scoped_unspecified_t InvalidInsideRange4 = static_cast<scoped_unspecified_t>(3); // expected-warning {{The value '3' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t ValidPositiveValue3 = static_cast<scoped_unspecified_t>(4); // OK.
  scoped_unspecified_t InvalidAfterRangeEnd = static_cast<scoped_unspecified_t>(5); // expected-warning {{The value '5' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
}

void scopedSpecified() {
  scoped_specified_t InvalidBeforeRangeBegin = static_cast<scoped_specified_t>(-5); // expected-warning {{The value '-5' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t ValidNegativeValue1 = static_cast<scoped_specified_t>(-4); // OK.
  scoped_specified_t ValidNegativeValue2 = static_cast<scoped_specified_t>(-3); // OK.
  scoped_specified_t InvalidInsideRange1 = static_cast<scoped_specified_t>(-2); // expected-warning {{The value '-2' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t InvalidInsideRange2 = static_cast<scoped_specified_t>(-1); // expected-warning {{The value '-1' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t InvalidInsideRange3 = static_cast<scoped_specified_t>(0); // expected-warning {{The value '0' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t ValidPositiveValue1 = static_cast<scoped_specified_t>(1); // OK.
  scoped_specified_t ValidPositiveValue2 = static_cast<scoped_specified_t>(2); // OK.
  scoped_specified_t InvalidInsideRange4 = static_cast<scoped_specified_t>(3); // expected-warning {{The value '3' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t ValidPositiveValue3 = static_cast<scoped_specified_t>(4); // OK.
  scoped_specified_t InvalidAfterRangeEnd = static_cast<scoped_specified_t>(5); // expected-warning {{The value '5' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
}

void unscopedUnspecifiedCStyle() {
  unscoped_unspecified_t InvalidBeforeRangeBegin = (unscoped_unspecified_t)(-5); // expected-warning {{The value '-5' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t ValidNegativeValue1 = (unscoped_unspecified_t)(-4); // OK.
  unscoped_unspecified_t ValidNegativeValue2 = (unscoped_unspecified_t)(-3); // OK.
  unscoped_unspecified_t InvalidInsideRange1 = (unscoped_unspecified_t)(-2); // expected-warning {{The value '-2' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t InvalidInsideRange2 = (unscoped_unspecified_t)(-1); // expected-warning {{The value '-1' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t InvalidInsideRange3 = (unscoped_unspecified_t)(0); // expected-warning {{The value '0' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t ValidPositiveValue1 = (unscoped_unspecified_t)(1); // OK.
  unscoped_unspecified_t ValidPositiveValue2 = (unscoped_unspecified_t)(2); // OK.
  unscoped_unspecified_t InvalidInsideRange4 = (unscoped_unspecified_t)(3); // expected-warning {{The value '3' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
  unscoped_unspecified_t ValidPositiveValue3 = (unscoped_unspecified_t)(4); // OK.
  unscoped_unspecified_t InvalidAfterRangeEnd = (unscoped_unspecified_t)(5); // expected-warning {{The value '5' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
}

void unscopedSpecifiedCStyle() {
  unscoped_specified_t InvalidBeforeRangeBegin = (unscoped_specified_t)(-5); // expected-warning {{The value '-5' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t ValidNegativeValue1 = (unscoped_specified_t)(-4); // OK.
  unscoped_specified_t ValidNegativeValue2 = (unscoped_specified_t)(-3); // OK.
  unscoped_specified_t InvalidInsideRange1 = (unscoped_specified_t)(-2); // expected-warning {{The value '-2' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t InvalidInsideRange2 = (unscoped_specified_t)(-1); // expected-warning {{The value '-1' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t InvalidInsideRange3 = (unscoped_specified_t)(0); // expected-warning {{The value '0' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t ValidPositiveValue1 = (unscoped_specified_t)(1); // OK.
  unscoped_specified_t ValidPositiveValue2 = (unscoped_specified_t)(2); // OK.
  unscoped_specified_t InvalidInsideRange4 = (unscoped_specified_t)(3); // expected-warning {{The value '3' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
  unscoped_specified_t ValidPositiveValue3 = (unscoped_specified_t)(4); // OK.
  unscoped_specified_t InvalidAfterRangeEnd = (unscoped_specified_t)(5); // expected-warning {{The value '5' provided to the cast expression is not in the valid range of values for 'unscoped_specified_t'}}
}

void scopedUnspecifiedCStyle() {
  scoped_unspecified_t InvalidBeforeRangeBegin = (scoped_unspecified_t)(-5); // expected-warning{{The value '-5' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t ValidNegativeValue1 = (scoped_unspecified_t)(-4); // OK.
  scoped_unspecified_t ValidNegativeValue2 = (scoped_unspecified_t)(-3); // OK.
  scoped_unspecified_t InvalidInsideRange1 = (scoped_unspecified_t)(-2); // expected-warning {{The value '-2' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t InvalidInsideRange2 = (scoped_unspecified_t)(-1); // expected-warning {{The value '-1' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t InvalidInsideRange3 = (scoped_unspecified_t)(0); // expected-warning {{The value '0' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t ValidPositiveValue1 = (scoped_unspecified_t)(1); // OK.
  scoped_unspecified_t ValidPositiveValue2 = (scoped_unspecified_t)(2); // OK.
  scoped_unspecified_t InvalidInsideRange4 = (scoped_unspecified_t)(3); // expected-warning {{The value '3' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
  scoped_unspecified_t ValidPositiveValue3 = (scoped_unspecified_t)(4); // OK.
  scoped_unspecified_t InvalidAfterRangeEnd = (scoped_unspecified_t)(5); // expected-warning {{The value '5' provided to the cast expression is not in the valid range of values for 'scoped_unspecified_t'}}
}

void scopedSpecifiedCStyle() {
  scoped_specified_t InvalidBeforeRangeBegin = (scoped_specified_t)(-5); // expected-warning {{The value '-5' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t ValidNegativeValue1 = (scoped_specified_t)(-4); // OK.
  scoped_specified_t ValidNegativeValue2 = (scoped_specified_t)(-3); // OK.
  scoped_specified_t InvalidInsideRange1 = (scoped_specified_t)(-2); // expected-warning {{The value '-2' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t InvalidInsideRange2 = (scoped_specified_t)(-1); // expected-warning {{The value '-1' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t InvalidInsideRange3 = (scoped_specified_t)(0); // expected-warning {{The value '0' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t ValidPositiveValue1 = (scoped_specified_t)(1); // OK.
  scoped_specified_t ValidPositiveValue2 = (scoped_specified_t)(2); // OK.
  scoped_specified_t InvalidInsideRange4 = (scoped_specified_t)(3); // expected-warning {{The value '3' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
  scoped_specified_t ValidPositiveValue3 = (scoped_specified_t)(4); // OK.
  scoped_specified_t InvalidAfterRangeEnd = (scoped_specified_t)(5); // expected-warning {{The value '5' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
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
    auto value = static_cast<scoped_specified_t>(input); // expected-warning {{The value provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
}

void rangeConstrained3(int input) {
  if (input >= -2 && input <= -1)
    auto value = static_cast<scoped_specified_t>(input); // expected-warning {{The value provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
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
    auto value = static_cast<scoped_specified_t>(input); // expected-warning {{The value '3' provided to the cast expression is not in the valid range of values for 'scoped_specified_t'}}
}

void enumBitFieldAssignment() {
  S s;
  s.E = static_cast<unscoped_unspecified_t>(4); // OK.
  s.E = static_cast<unscoped_unspecified_t>(5); // expected-warning {{The value '5' provided to the cast expression is not in the valid range of values for 'unscoped_unspecified_t'}}
}


enum class empty_unspecified {};

enum class empty_specified: char {};

enum class empty_specified_unsigned: unsigned char {};

void ignore_unused(...);

void empty_enums_init_with_zero_should_not_warn() {
  auto eu = static_cast<empty_unspecified>(0); //should always be OK to zero initialize any enum
  auto ef = static_cast<empty_specified>(0);
  auto efu = static_cast<empty_specified_unsigned>(0);

  ignore_unused(eu, ef, efu);
}

//Test the example from checkers.rst:
enum WidgetKind { A=1, B, C, X=99 }; // expected-note {{enum declared here}}

void foo() {
  WidgetKind c = static_cast<WidgetKind>(3);  // OK
  WidgetKind x = static_cast<WidgetKind>(99); // OK
  WidgetKind d = static_cast<WidgetKind>(4);  // expected-warning {{The value '4' provided to the cast expression is not in the valid range of values for 'WidgetKind'}}

  ignore_unused(c, x, d);
}

enum [[clang::flag_enum]] FlagEnum {
  FE_BIT_1 = 1 << 0,
  FE_BIT_2 = 1 << 1,
  FE_BIT_3 = 1 << 2,
};

void testFlagEnum_gh_76208(void) {
  FlagEnum First2BitsSet = (FlagEnum)(FE_BIT_1 | FE_BIT_2); // no-warning: Enums with the attribute 'flag_enum' are not checked
  (void)First2BitsSet;
}
