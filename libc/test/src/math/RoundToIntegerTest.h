//===-- Utility class to test different flavors of [l|ll]round --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_ROUNDTOINTEGERTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_ROUNDTOINTEGERTEST_H

#include "utils/FPUtil/FPBits.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

#include <math.h>
#if math_errhandling & MATH_ERRNO
#include <errno.h>
#endif
#if math_errhandling & MATH_ERREXCEPT
#include "utils/FPUtil/FEnv.h"
#endif

namespace mpfr = __llvm_libc::testing::mpfr;

static constexpr int roundingModes[4] = {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO,
                                         FE_TONEAREST};

template <typename F, typename I, bool TestModes = false>
class RoundToIntegerTestTemplate : public __llvm_libc::testing::Test {
public:
  typedef I (*RoundToIntegerFunc)(F);

private:
  using FPBits = __llvm_libc::fputil::FPBits<F>;
  using UIntType = typename FPBits::UIntType;

  const F zero = __llvm_libc::fputil::FPBits<F>::zero();
  const F negZero = __llvm_libc::fputil::FPBits<F>::negZero();
  const F inf = __llvm_libc::fputil::FPBits<F>::inf();
  const F negInf = __llvm_libc::fputil::FPBits<F>::negInf();
  const F nan = __llvm_libc::fputil::FPBits<F>::buildNaN(1);
  static constexpr I IntegerMin = I(1) << (sizeof(I) * 8 - 1);
  static constexpr I IntegerMax = -(IntegerMin + 1);

  void testOneInput(RoundToIntegerFunc func, F input, I expected,
                    bool expectError) {
#if math_errhandling & MATH_ERRNO
    errno = 0;
#endif
#if math_errhandling & MATH_ERREXCEPT
    __llvm_libc::fputil::clearExcept(FE_ALL_EXCEPT);
#endif

    ASSERT_EQ(func(input), expected);

    if (expectError) {
#if math_errhandling & MATH_ERREXCEPT
      ASSERT_EQ(__llvm_libc::fputil::testExcept(FE_ALL_EXCEPT), FE_INVALID);
#endif
#if math_errhandling & MATH_ERRNO
      ASSERT_EQ(errno, EDOM);
#endif
    } else {
#if math_errhandling & MATH_ERREXCEPT
      ASSERT_EQ(__llvm_libc::fputil::testExcept(FE_ALL_EXCEPT), 0);
#endif
#if math_errhandling & MATH_ERRNO
      ASSERT_EQ(errno, 0);
#endif
    }
  }

  static inline mpfr::RoundingMode toMPFRRoundingMode(int mode) {
    switch (mode) {
    case FE_UPWARD:
      return mpfr::RoundingMode::Upward;
    case FE_DOWNWARD:
      return mpfr::RoundingMode::Downward;
    case FE_TOWARDZERO:
      return mpfr::RoundingMode::TowardZero;
    case FE_TONEAREST:
      return mpfr::RoundingMode::Nearest;
    default:
      __builtin_unreachable();
    }
  }

public:
  void SetUp() override {
#if math_errhandling & MATH_ERREXCEPT
    // We will disable all exceptions so that the test will not
    // crash with SIGFPE. We can still use fetestexcept to check
    // if the appropriate flag was raised.
    __llvm_libc::fputil::disableExcept(FE_ALL_EXCEPT);
#endif
  }

  void doInfinityAndNaNTest(RoundToIntegerFunc func) {
    testOneInput(func, inf, IntegerMax, true);
    testOneInput(func, negInf, IntegerMin, true);
    testOneInput(func, nan, IntegerMax, true);
  }

  void testInfinityAndNaN(RoundToIntegerFunc func) {
    if (TestModes) {
      for (int mode : roundingModes) {
        __llvm_libc::fputil::setRound(mode);
        doInfinityAndNaNTest(func);
      }
    } else {
      doInfinityAndNaNTest(func);
    }
  }

  void doRoundNumbersTest(RoundToIntegerFunc func) {
    testOneInput(func, zero, I(0), false);
    testOneInput(func, negZero, I(0), false);
    testOneInput(func, F(1.0), I(1), false);
    testOneInput(func, F(-1.0), I(-1), false);
    testOneInput(func, F(10.0), I(10), false);
    testOneInput(func, F(-10.0), I(-10), false);
    testOneInput(func, F(1234.0), I(1234), false);
    testOneInput(func, F(-1234.0), I(-1234), false);

    // The rest of this this function compares with an equivalent MPFR function
    // which rounds floating point numbers to long values. There is no MPFR
    // function to round to long long or wider integer values. So, we will
    // the remaining tests only if the width of I less than equal to that of
    // long.
    if (sizeof(I) > sizeof(long))
      return;

    constexpr int exponentLimit = sizeof(I) * 8 - 1;
    // We start with 1.0 so that the implicit bit for x86 long doubles
    // is set.
    FPBits bits(F(1.0));
    bits.exponent = exponentLimit + FPBits::exponentBias;
    bits.sign = 1;
    bits.mantissa = 0;

    F x = bits;
    long mpfrResult;
    bool erangeflag = mpfr::RoundToLong(x, mpfrResult);
    ASSERT_FALSE(erangeflag);
    testOneInput(func, x, mpfrResult, false);
  }

  void testRoundNumbers(RoundToIntegerFunc func) {
    if (TestModes) {
      for (int mode : roundingModes) {
        __llvm_libc::fputil::setRound(mode);
        doRoundNumbersTest(func);
      }
    } else {
      doRoundNumbersTest(func);
    }
  }

  void doFractionsTest(RoundToIntegerFunc func, int mode) {
    constexpr F fractions[] = {0.5, -0.5, 0.115, -0.115, 0.715, -0.715};
    for (F x : fractions) {
      long mpfrLongResult;
      bool erangeflag;
      if (TestModes)
        erangeflag =
            mpfr::RoundToLong(x, toMPFRRoundingMode(mode), mpfrLongResult);
      else
        erangeflag = mpfr::RoundToLong(x, mpfrLongResult);
      ASSERT_FALSE(erangeflag);
      I mpfrResult = mpfrLongResult;
      testOneInput(func, x, mpfrResult, false);
    }
  }

  void testFractions(RoundToIntegerFunc func) {
    if (TestModes) {
      for (int mode : roundingModes) {
        __llvm_libc::fputil::setRound(mode);
        doFractionsTest(func, mode);
      }
    } else {
      // Passing 0 for mode has no effect as it is not used in doFractionsTest
      // when `TestModes` is false;
      doFractionsTest(func, 0);
    }
  }

  void testIntegerOverflow(RoundToIntegerFunc func) {
    // This function compares with an equivalent MPFR function which rounds
    // floating point numbers to long values. There is no MPFR function to
    // round to long long or wider integer values. So, we will peform the
    // comparisons in this function only if the width of I less than equal to
    // that of long.
    if (sizeof(I) > sizeof(long))
      return;

    constexpr int exponentLimit = sizeof(I) * 8 - 1;
    // We start with 1.0 so that the implicit bit for x86 long doubles
    // is set.
    FPBits bits(F(1.0));
    bits.exponent = exponentLimit + FPBits::exponentBias;
    bits.sign = 1;
    bits.mantissa = UIntType(0x1)
                    << (__llvm_libc::fputil::MantissaWidth<F>::value - 1);

    F x = bits;
    if (TestModes) {
      for (int m : roundingModes) {
        __llvm_libc::fputil::setRound(m);
        long mpfrLongResult;
        bool erangeflag =
            mpfr::RoundToLong(x, toMPFRRoundingMode(m), mpfrLongResult);
        ASSERT_TRUE(erangeflag);
        testOneInput(func, x, IntegerMin, true);
      }
    } else {
      long mpfrLongResult;
      bool erangeflag = mpfr::RoundToLong(x, mpfrLongResult);
      ASSERT_TRUE(erangeflag);
      testOneInput(func, x, IntegerMin, true);
    }
  }

  void testSubnormalRange(RoundToIntegerFunc func) {
    constexpr UIntType count = 1000001;
    constexpr UIntType step =
        (FPBits::maxSubnormal - FPBits::minSubnormal) / count;
    for (UIntType i = FPBits::minSubnormal; i <= FPBits::maxSubnormal;
         i += step) {
      F x = FPBits(i);
      if (x == F(0.0))
        continue;
      // All subnormal numbers should round to zero.
      if (TestModes) {
        if (x > 0) {
          __llvm_libc::fputil::setRound(FE_UPWARD);
          testOneInput(func, x, I(1), false);
          __llvm_libc::fputil::setRound(FE_DOWNWARD);
          testOneInput(func, x, I(0), false);
          __llvm_libc::fputil::setRound(FE_TOWARDZERO);
          testOneInput(func, x, I(0), false);
          __llvm_libc::fputil::setRound(FE_TONEAREST);
          testOneInput(func, x, I(0), false);
        } else {
          __llvm_libc::fputil::setRound(FE_UPWARD);
          testOneInput(func, x, I(0), false);
          __llvm_libc::fputil::setRound(FE_DOWNWARD);
          testOneInput(func, x, I(-1), false);
          __llvm_libc::fputil::setRound(FE_TOWARDZERO);
          testOneInput(func, x, I(0), false);
          __llvm_libc::fputil::setRound(FE_TONEAREST);
          testOneInput(func, x, I(0), false);
        }
      } else {
        testOneInput(func, x, 0L, false);
      }
    }
  }

  void testNormalRange(RoundToIntegerFunc func) {
    // This function compares with an equivalent MPFR function which rounds
    // floating point numbers to long values. There is no MPFR function to
    // round to long long or wider integer values. So, we will peform the
    // comparisons in this function only if the width of I less than equal to
    // that of long.
    if (sizeof(I) > sizeof(long))
      return;

    constexpr UIntType count = 1000001;
    constexpr UIntType step = (FPBits::maxNormal - FPBits::minNormal) / count;
    for (UIntType i = FPBits::minNormal; i <= FPBits::maxNormal; i += step) {
      F x = FPBits(i);
      // In normal range on x86 platforms, the long double implicit 1 bit can be
      // zero making the numbers NaN. We will skip them.
      if (isnan(x)) {
        continue;
      }

      if (TestModes) {
        for (int m : roundingModes) {
          long mpfrLongResult;
          bool erangeflag =
              mpfr::RoundToLong(x, toMPFRRoundingMode(m), mpfrLongResult);
          I mpfrResult = mpfrLongResult;
          __llvm_libc::fputil::setRound(m);
          if (erangeflag)
            testOneInput(func, x, x > 0 ? IntegerMax : IntegerMin, true);
          else
            testOneInput(func, x, mpfrResult, false);
        }
      } else {
        long mpfrLongResult;
        bool erangeflag = mpfr::RoundToLong(x, mpfrLongResult);
        I mpfrResult = mpfrLongResult;
        if (erangeflag)
          testOneInput(func, x, x > 0 ? IntegerMax : IntegerMin, true);
        else
          testOneInput(func, x, mpfrResult, false);
      }
    }
  }
};

#define LIST_ROUND_TO_INTEGER_TESTS_HELPER(F, I, func, TestModes)              \
  using LlvmLibcRoundToIntegerTest =                                           \
      RoundToIntegerTestTemplate<F, I, TestModes>;                             \
  TEST_F(LlvmLibcRoundToIntegerTest, InfinityAndNaN) {                         \
    testInfinityAndNaN(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibcRoundToIntegerTest, RoundNumbers) {                           \
    testRoundNumbers(&func);                                                   \
  }                                                                            \
  TEST_F(LlvmLibcRoundToIntegerTest, Fractions) { testFractions(&func); }      \
  TEST_F(LlvmLibcRoundToIntegerTest, IntegerOverflow) {                        \
    testIntegerOverflow(&func);                                                \
  }                                                                            \
  TEST_F(LlvmLibcRoundToIntegerTest, SubnormalRange) {                         \
    testSubnormalRange(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibcRoundToIntegerTest, NormalRange) { testNormalRange(&func); }

#define LIST_ROUND_TO_INTEGER_TESTS(F, I, func)                                \
  LIST_ROUND_TO_INTEGER_TESTS_HELPER(F, I, func, false)

#define LIST_ROUND_TO_INTEGER_TESTS_WITH_MODES(F, I, func)                     \
  LIST_ROUND_TO_INTEGER_TESTS_HELPER(F, I, func, true)

#endif // LLVM_LIBC_TEST_SRC_MATH_ROUNDTOINTEGERTEST_H
