//===-- Utility class to test different flavors of [l|ll]round --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_ROUNDTOINTEGERTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_ROUNDTOINTEGERTEST_H

#include "src/errno/llvmlibc_errno.h"
#include "src/fenv/feclearexcept.h"
#include "src/fenv/feraiseexcept.h"
#include "src/fenv/fetestexcept.h"
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

template <typename F, typename I>
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
    llvmlibc_errno = 0;
#endif
#if math_errhandling & MATH_ERREXCEPT
    __llvm_libc::feclearexcept(FE_ALL_EXCEPT);
#endif

    ASSERT_EQ(func(input), expected);

    if (expectError) {
#if math_errhandling & MATH_ERREXCEPT
      ASSERT_EQ(__llvm_libc::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
#endif
#if math_errhandling & MATH_ERRNO
      ASSERT_EQ(llvmlibc_errno, EDOM);
#endif
    } else {
#if math_errhandling & MATH_ERREXCEPT
      ASSERT_EQ(__llvm_libc::fetestexcept(FE_ALL_EXCEPT), 0);
#endif
#if math_errhandling & MATH_ERRNO
      ASSERT_EQ(llvmlibc_errno, 0);
#endif
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

  void testInfinityAndNaN(RoundToIntegerFunc func) {
    testOneInput(func, inf, IntegerMax, true);
    testOneInput(func, negInf, IntegerMin, true);
    testOneInput(func, nan, IntegerMax, true);
  }

  void testRoundNumbers(RoundToIntegerFunc func) {
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

  void testFractions(RoundToIntegerFunc func) {
    testOneInput(func, F(0.5), I(1), false);
    testOneInput(func, F(-0.5), I(-1), false);
    testOneInput(func, F(0.115), I(0), false);
    testOneInput(func, F(-0.115), I(0), false);
    testOneInput(func, F(0.715), I(1), false);
    testOneInput(func, F(-0.715), I(-1), false);
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
    long mpfrResult;
    bool erangeflag = mpfr::RoundToLong(x, mpfrResult);
    ASSERT_TRUE(erangeflag);
    testOneInput(func, x, IntegerMin, true);
  }

  void testSubnormalRange(RoundToIntegerFunc func) {
    // This function compares with an equivalent MPFR function which rounds
    // floating point numbers to long values. There is no MPFR function to
    // round to long long or wider integer values. So, we will peform the
    // comparisons in this function only if the width of I less than equal to
    // that of long.
    if (sizeof(I) > sizeof(long))
      return;

    constexpr UIntType count = 1000001;
    constexpr UIntType step =
        (FPBits::maxSubnormal - FPBits::minSubnormal) / count;
    for (UIntType i = FPBits::minSubnormal; i <= FPBits::maxSubnormal;
         i += step) {
      F x = FPBits(i);
      // All subnormal numbers should round to zero.
      testOneInput(func, x, 0L, false);
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

      long mpfrResult;
      bool erangeflag = mpfr::RoundToLong(x, mpfrResult);
      if (erangeflag)
        testOneInput(func, x, x > 0 ? IntegerMax : IntegerMin, true);
      else
        testOneInput(func, x, mpfrResult, false);
    }
  }
};

#define LIST_ROUND_TO_INTEGER_TESTS(F, I, func)                                \
  using RoundToIntegerTest = RoundToIntegerTestTemplate<F, I>;                 \
  TEST_F(RoundToIntegerTest, InfinityAndNaN) { testInfinityAndNaN(&func); }    \
  TEST_F(RoundToIntegerTest, RoundNumbers) { testRoundNumbers(&func); }        \
  TEST_F(RoundToIntegerTest, Fractions) { testFractions(&func); }              \
  TEST_F(RoundToIntegerTest, IntegerOverflow) { testIntegerOverflow(&func); }  \
  TEST_F(RoundToIntegerTest, SubnormalRange) { testSubnormalRange(&func); }    \
  TEST_F(RoundToIntegerTest, NormalRange) { testNormalRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_ROUNDTOINTEGERTEST_H
