// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-integer-sign-comparison %t

// CHECK-FIXES: #include <utility>

namespace std {
template <class T>
struct numeric_limits {
  static constexpr T min() noexcept;
  static constexpr T lowest() noexcept;
  static constexpr T max() noexcept;
};
} // namespace std

// The code that triggers the check
#define MAX_MACRO(a, b) (a < b) ? b : a

unsigned int FuncParameters(int bla) {
    unsigned int result = 0;
    if (result == bla)
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_equal(result , bla))

    return 1;
}

template <typename T>
void TemplateFuncParameter(T val) {
    unsigned long uL = 0;
    if (val >= uL)
        return;
// CHECK-MESSAGES-NOT: warning:
}

template <typename T1, typename T2>
int TemplateFuncParameters(T1 val1, T2 val2) {
    if (val1 >= val2)
        return 0;
// CHECK-MESSAGES-NOT: warning:
    return 1;
}

int AllComparisons() {
    unsigned int uVar = 42;
    unsigned short uArray[7] = {0, 1, 2, 3, 9, 7, 9};

    int sVar = -42;
    short sArray[7] = {-1, -2, -8, -94, -5, -4, -6};

    enum INT_TEST {
      VAL1 = 0,
      VAL2 = -1
    };

    char ch = 'a';
    unsigned char uCh = 'a';
    signed char sCh = 'a';
    bool bln = false;

    if (bln == sVar)
      return 0;
// CHECK-MESSAGES-NOT: warning:

    if (ch > uCh)
      return 0;
// CHECK-MESSAGES-NOT: warning:

    if (sVar <= INT_TEST::VAL2)
      return 0;
// CHECK-MESSAGES-NOT: warning:

    if (uCh < sCh)
      return -1;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_less(uCh , sCh))

    if ((int)uVar < sVar)
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_less(uVar, sVar))

    (uVar != sVar) ? uVar = sVar
                   : sVar = uVar;
// CHECK-MESSAGES: :[[@LINE-2]]:6: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: (std::cmp_not_equal(uVar , sVar)) ? uVar = sVar

    while (uArray[0] <= sArray[0])
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:12: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: while (std::cmp_less_equal(uArray[0] , sArray[0]))

    if (uArray[1] > sArray[1])
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_greater(uArray[1] , sArray[1]))

    MAX_MACRO(uVar, sArray[0]);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]

    if (static_cast<unsigned int>(uArray[2]) < static_cast<int>(sArray[2]))
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_less(uArray[2],sArray[2]))

    if ((unsigned int)uArray[3] < (int)sArray[3])
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_less(uArray[3],sArray[3]))

    if ((unsigned int)(uArray[4]) < (int)(sArray[4]))
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_less((uArray[4]),(sArray[4])))

    if (uArray[5] > sArray[5])
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_greater(uArray[5] , sArray[5]))

    #define VALUE sArray[6]
    if (uArray[6] > VALUE)
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_greater(uArray[6] , VALUE))

    if (unsigned(uArray[7]) >= int(sArray[7]))
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (std::cmp_greater_equal(uArray[7],sArray[7]))


    FuncParameters(uVar);
    TemplateFuncParameter(sVar);
    TemplateFuncParameters(uVar, sVar);

    return 0;
}

// ---- in_range: positive tests ----

bool canonical(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range' instead of manual range check [modernize-use-integer-sign-comparison]
  // CHECK-FIXES: return std::in_range<int>(val);
  return val >= std::numeric_limits<int>::min() && val <= std::numeric_limits<int>::max();
}

bool commutative_min_lhs(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<int>(val);
  return std::numeric_limits<int>::min() <= val && val <= std::numeric_limits<int>::max();
}

bool commutative_max_lhs(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<int>(val);
  return val >= std::numeric_limits<int>::min() && std::numeric_limits<int>::max() >= val;
}

bool swapped_and_operands(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<int>(val);
  return val <= std::numeric_limits<int>::max() && val >= std::numeric_limits<int>::min();
}

bool with_lowest(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<int>(val);
  return val >= std::numeric_limits<int>::lowest() && val <= std::numeric_limits<int>::max();
}

bool negated_form(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<int>(val);
  return !(val < std::numeric_limits<int>::min() || val > std::numeric_limits<int>::max());
}

bool negated_swapped_or(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<int>(val);
  return !(val > std::numeric_limits<int>::max() || val < std::numeric_limits<int>::min());
}

bool short_type(int val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<short>(val);
  return val >= std::numeric_limits<short>::min() && val <= std::numeric_limits<short>::max();
}

// typedef aliases must be preserved in the fix-it
using int32_t = int;
using uint16_t = unsigned short;

bool typedef_signed(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<int32_t>(val);
  return val >= std::numeric_limits<int32_t>::min() && val <= std::numeric_limits<int32_t>::max();
}

bool long_long_type(long long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<int>(val);
  return val >= std::numeric_limits<int>::min() && val <= std::numeric_limits<int>::max();
}

bool signed_char_type(int val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<signed char>(val);
  return val >= std::numeric_limits<signed char>::min() && val <= std::numeric_limits<signed char>::max();
}

template <typename T>
bool template_value(T val) {
  // CHECK-MESSAGES-NOT: warning:
  return val >= std::numeric_limits<int>::min() && val <= std::numeric_limits<int>::max();
}
bool instantiate_template() { return template_value(42L); }

// ---- in_range: negative tests (should NOT trigger) ----

bool mismatched_types(long val) {
  // CHECK-MESSAGES-NOT: warning:
  return val >= std::numeric_limits<int>::min() && val <= std::numeric_limits<short>::max();
}

bool different_values(long val1, long val2) {
  // CHECK-MESSAGES-NOT: warning:
  return val1 >= std::numeric_limits<int>::min() && val2 <= std::numeric_limits<int>::max();
}

bool float_type(double val) {
  // CHECK-MESSAGES-NOT: warning:
  return val >= std::numeric_limits<float>::min() && val <= std::numeric_limits<float>::max();
}

bool bool_type(int val) {
  // CHECK-MESSAGES-NOT: warning:
  return val >= std::numeric_limits<bool>::min() && val <= std::numeric_limits<bool>::max();
}

bool char_type(int val) {
  // CHECK-MESSAGES-NOT: warning:
  return val >= std::numeric_limits<char>::min() && val <= std::numeric_limits<char>::max();
}

enum MyEnum { A, B };
bool enum_type(int val) {
  // CHECK-MESSAGES-NOT: warning:
  return val >= std::numeric_limits<MyEnum>::min() && val <= std::numeric_limits<MyEnum>::max();
}

bool wrong_operators(long val) {
  // CHECK-MESSAGES-NOT: warning:
  return val >= std::numeric_limits<int>::min() && val >= std::numeric_limits<int>::max();
}

// Side-effecting value expression: should NOT trigger (unsafe to collapse two
// calls into one, since doing so would change the number of calls).
int counter();
bool side_effects(long) {
  // CHECK-MESSAGES-NOT: warning:
  return counter() >= std::numeric_limits<int>::min() && counter() <= std::numeric_limits<int>::max();
}

// Range check inside a macro body: diagnostic fires but no fix-it (the
// replacement range spans a macro expansion).
#define FITS_IN_INT(v) \
  (v) >= std::numeric_limits<int>::min() && (v) <= std::numeric_limits<int>::max()

bool in_macro_body(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:{{.*}}warning: use 'std::in_range'
  // CHECK-FIXES-NOT: std::in_range
  return FITS_IN_INT(val);
}

// ---- cmp_* vs in_range interaction ----

// When val and T have opposite signedness (the canonical in_range use case),
// the individual sub-comparisons are themselves signed/unsigned comparisons
// that would trigger cmp_* warnings. The in_range pattern takes priority:
// only the in_range diagnostic should fire, not cmp_* for each sub-expression.
//
// long vs unsigned long: same bit-width, so the usual arithmetic conversion
// promotes long to unsigned long -- a genuine signed/unsigned comparison.
bool fits_in_ulong(long val) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<unsigned long>(val);
  return val >= std::numeric_limits<unsigned long>::min() && val <= std::numeric_limits<unsigned long>::max();
}
// The two sub-comparisons above must NOT produce cmp_* warnings;
// they are suppressed by the two-pass range-check filter in onEndOfTranslationUnit().

// Fix 1: a standalone signed/unsigned comparison that involves a numeric_limits
// call but is NOT part of a range check must still warn with cmp_*.
// uval (unsigned long) > min() (int): the int return value is implicitly cast
// to unsigned long, so CompareOperator fires and the warning must not be
// suppressed just because one operand is a numeric_limits call.
bool standalone_limits_cmp(unsigned long uval) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: comparison between 'signed' and 'unsigned' integers
  // CHECK-FIXES: return std::cmp_greater(uval , std::numeric_limits<int>::min());
  return uval > std::numeric_limits<int>::min();
}

// Fix 3: object-style call lim.min() must preserve the typedef alias in the
// fix-it via the MemberExpr callee path in getLimitsTypeSourceText.
bool object_style_limits(long val) {
  std::numeric_limits<int32_t> lim;
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::in_range'
  // CHECK-FIXES: return std::in_range<int32_t>(val);
  return val >= lim.std::numeric_limits<int32_t>::min() && val <= lim.std::numeric_limits<int32_t>::max();
}
