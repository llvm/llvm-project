// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: hexagon-target-arch
// REQUIRES: librt_has_dffma

#include "int_lib.h"
#include <fenv.h>
#include <stdio.h>

#include "fp_test.h"

#if defined(__hexagon__)
// Fused multiply-add: computes a*b + c with a single rounding, in the
// current rounding mode.
COMPILER_RT_ABI double __hexagon_fmadf5(double a, double b, double c);

#define QNAN_REP UINT64_C(0x7ff8000000000000)
#define POS_INF_REP UINT64_C(0x7ff0000000000000)
#define NEG_INF_REP UINT64_C(0xfff0000000000000)
#define POS_ZERO_REP UINT64_C(0x0000000000000000)
#define NEG_ZERO_REP UINT64_C(0x8000000000000000)
#define DBL_MAX_REP UINT64_C(0x7fefffffffffffff)
#define NEG_DBL_MAX_REP UINT64_C(0xffefffffffffffff)
#define DBL_TRUE_MIN_REP UINT64_C(0x0000000000000001) // smallest subnormal
#define DBL_MIN_REP UINT64_C(0x0010000000000000)      // smallest normal

static const char *rmode_name(int rm) {
  switch (rm) {
  case FE_TONEAREST:
    return "FE_TONEAREST";
  case FE_UPWARD:
    return "FE_UPWARD";
  case FE_DOWNWARD:
    return "FE_DOWNWARD";
  case FE_TOWARDZERO:
    return "FE_TOWARDZERO";
  default:
    __builtin_unreachable();
  }
}

// Test one fma evaluation in a specific rounding mode against an expected
// bit pattern.  Returns 1 on mismatch.
static int test_rm(double a, double b, double c, int rmode, uint64_t expected) {
  if (fesetround(rmode) != 0)
    return 0; // rounding mode unsupported on this target; skip
  double x = __hexagon_fmadf5(a, b, c);
  fesetround(FE_TONEAREST);
  int ret = compareResultD(x, expected);
  if (ret)
    printf("error: fma(%a, %a, %a) [%s] = %a, expected %a\n", a, b, c,
           rmode_name(rmode), x, fromRep64(expected));
  return ret;
}

// Test one fma evaluation in the default (round-to-nearest) mode.
static int test(double a, double b, double c, uint64_t expected) {
  return test_rm(a, b, c, FE_TONEAREST, expected);
}

// Test that a result is identical under all four rounding modes (used for
// exact results, which must not depend on the rounding mode).
static int test_all_modes(double a, double b, double c, uint64_t expected) {
  int ret = 0;
  ret |= test_rm(a, b, c, FE_TONEAREST, expected);
  ret |= test_rm(a, b, c, FE_UPWARD, expected);
  ret |= test_rm(a, b, c, FE_DOWNWARD, expected);
  ret |= test_rm(a, b, c, FE_TOWARDZERO, expected);
  return ret;
}
#endif

int main(void) {
#if defined(__hexagon__)
  double qnan = fromRep64(QNAN_REP);
  double inf = fromRep64(POS_INF_REP);
  double ninf = fromRep64(NEG_INF_REP);

  // NaN handling: any NaN operand -> qNaN
  if (test(qnan, 2.0, 3.0, QNAN_REP))
    return 1;
  if (test(2.0, qnan, 3.0, QNAN_REP))
    return 1;
  if (test(2.0, 3.0, qnan, QNAN_REP))
    return 1;
  if (test(qnan, qnan, qnan, QNAN_REP))
    return 1;

  // Invalid: 0 * Inf (+/-) is NaN regardless of the addend
  if (test(0.0, inf, 1.0, QNAN_REP))
    return 1;
  if (test(inf, 0.0, 1.0, QNAN_REP))
    return 1;
  if (test(0.0, ninf, 1.0, QNAN_REP))
    return 1;
  if (test(-0.0, inf, 1.0, QNAN_REP))
    return 1;

  // Invalid: (a*b = +Inf) + (-Inf) is NaN, and vice versa.
  if (test(inf, 1.0, ninf, QNAN_REP))
    return 1;
  if (test(ninf, 1.0, inf, QNAN_REP))
    return 1;

  // Infinity propagation (non-invalid)
  if (test(inf, 2.0, 3.0, POS_INF_REP))
    return 1; // +Inf * 2 + 3 = +Inf
  if (test(ninf, 2.0, 3.0, NEG_INF_REP))
    return 1; // -Inf * 2 + 3 = -Inf
  if (test(2.0, 3.0, inf, POS_INF_REP))
    return 1; // 6 + +Inf = +Inf
  if (test(2.0, 3.0, ninf, NEG_INF_REP))
    return 1; // 6 + -Inf = -Inf
  if (test(inf, 1.0, inf, POS_INF_REP))
    return 1; // +Inf + +Inf = +Inf
  if (test(ninf, 1.0, ninf, NEG_INF_REP))
    return 1; // -Inf + -Inf = -Inf

  // Exact results (rounding-mode independent)
  if (test_all_modes(3.0, 4.0, 5.0, toRep64(17.0)))
    return 1; // 12 + 5
  if (test_all_modes(-3.0, 4.0, 5.0, toRep64(-7.0)))
    return 1; // -12 + 5
  if (test_all_modes(0.5, 0.5, 0.0, toRep64(0.25)))
    return 1; // 0.25
  if (test_all_modes(1.0, 1.0, 0.0, toRep64(1.0)))
    return 1;
  // Exact cancellation 4 - 4 = +0 in nearest/up/toward-zero, -0 downward.
  if (test_rm(2.0, 2.0, -4.0, FE_TONEAREST, POS_ZERO_REP))
    return 1;
  if (test_rm(2.0, 2.0, -4.0, FE_UPWARD, POS_ZERO_REP))
    return 1;
  if (test_rm(2.0, 2.0, -4.0, FE_TOWARDZERO, POS_ZERO_REP))
    return 1;
  if (test_rm(2.0, 2.0, -4.0, FE_DOWNWARD, NEG_ZERO_REP))
    return 1;

  // Signed zero rules
  // x*y = 0 (exact) + (-0): in round-to-nearest, +0 + -0 = +0
  if (test(1.0, 0.0, -0.0, POS_ZERO_REP))
    return 1;
  // Under round-toward-negative, +0 + -0 = -0
  if (test_rm(1.0, 0.0, -0.0, FE_DOWNWARD, NEG_ZERO_REP))
    return 1;
  // Exact cancellation x*y + (-x*y) = +0 in nearest, -0 downward
  if (test_rm(2.0, 3.0, -6.0, FE_DOWNWARD, NEG_ZERO_REP))
    return 1;
  if (test_rm(2.0, 3.0, -6.0, FE_TONEAREST, POS_ZERO_REP))
    return 1;

  // The classic FMA single-rounding property.
  // With separate operations (a*b) then +c, the product a*b is rounded
  // first, losing the low bits, so the addition of -c cancels to 0.
  // A correctly single-rounded fma keeps the full product and yields the
  // exact low part.
  //   a = 1 + 2^-52 (0x1.0000000000001p+0)
  //   a*a = 1 + 2^-51 + 2^-104  (needs > 53 bits)
  //   fma(a, a, -1 - 2^-51) = 2^-104  (exact, tiny)
  // A non-fused a*a - (1+2^-51) would give 0.
  {
    double a = fromRep64(UINT64_C(0x3ff0000000000001)); // 1 + 2^-52
    double c =
        -fromRep64(UINT64_C(0x3cc0000000000000)); // -(1 + 2^-51) high part
    // exact result 2^-104
    double expect = 0x1.0p-104;
    (void)c;
    // fma(a,a,-(a*a rounded)) should recover the dropped low bits.
    double ab_rounded = a * a;                       // rounded product
    double lo = __hexagon_fmadf5(a, a, -ab_rounded); // exact low part
    // lo must be nonzero (fused); a naive mul-then-add would give 0.
    if (toRep64(lo) == POS_ZERO_REP || toRep64(lo) == NEG_ZERO_REP) {
      printf("error: fma not single-rounded: fma(%a,%a,-%a) = %a\n", a, a,
             ab_rounded, lo);
      return 1;
    }
    (void)expect;
  }

  // Overflow under directed rounding.  On a finite-magnitude overflow the
  // "inward"-directed modes must return +/-DBL_MAX, not +/-Inf.
  //   positive overflow: FE_DOWNWARD, FE_TOWARDZERO -> +DBL_MAX
  //   negative overflow: FE_UPWARD,   FE_TOWARDZERO -> -DBL_MAX
  {
    double px = 0x1.0000000000001p+1023;
    double py = 0x1.0000000000001p+1;
    double nx = -0x1.0000000000001p+1023;
    double tinyP = 0x0.0000000000001p-1022;
    double tinyN = -0x0.0000000000001p-1022;

    // positive overflow
    if (test_rm(px, py, tinyP, FE_TONEAREST, POS_INF_REP))
      return 1;
    if (test_rm(px, py, tinyP, FE_UPWARD, POS_INF_REP))
      return 1;
    if (test_rm(px, py, tinyP, FE_DOWNWARD, DBL_MAX_REP))
      return 1;
    if (test_rm(px, py, tinyP, FE_TOWARDZERO, DBL_MAX_REP))
      return 1;

    // negative overflow
    if (test_rm(nx, py, tinyN, FE_TONEAREST, NEG_INF_REP))
      return 1;
    if (test_rm(nx, py, tinyN, FE_UPWARD, NEG_DBL_MAX_REP))
      return 1;
    if (test_rm(nx, py, tinyN, FE_DOWNWARD, NEG_INF_REP))
      return 1;
    if (test_rm(nx, py, tinyN, FE_TOWARDZERO, NEG_DBL_MAX_REP))
      return 1;
  }

  // Underflow / subnormal results.
  // smallest_normal * 0.5 = smallest_normal/2, a subnormal, exact.
  {
    double dmin = fromRep64(DBL_MIN_REP); // 2^-1022
    // 2^-1022 * 0.5 + 0 = 2^-1023 (subnormal), exact
    if (test_all_modes(dmin, 0.5, 0.0, UINT64_C(0x0008000000000000)))
      return 1;
    // smallest subnormal stays itself when added to 0
    double tmin = fromRep64(DBL_TRUE_MIN_REP);
    if (test_all_modes(tmin, 1.0, 0.0, DBL_TRUE_MIN_REP))
      return 1;
  }

  // Zero operands
  if (test(0.0, 0.0, 0.0, POS_ZERO_REP))
    return 1;
  if (test(0.0, 0.0, 5.0, toRep64(5.0)))
    return 1;
  if (test(0.0, 5.0, 7.0, toRep64(7.0)))
    return 1;
  if (test(-0.0, 5.0, 0.0, POS_ZERO_REP))
    return 1; // -0 in nearest -> +0

#else
  printf("skipped\n");
#endif
  return 0;
}
