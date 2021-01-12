//===-- Utility class to test different flavors of fma --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_FMATEST_H
#define LLVM_LIBC_TEST_SRC_MATH_FMATEST_H

#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"
#include "utils/testutils/RandUtils.h"

namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T>
class FmaTestTemplate : public __llvm_libc::testing::Test {
private:
  using Func = T (*)(T, T, T);
  using FPBits = __llvm_libc::fputil::FPBits<T>;
  using UIntType = typename FPBits::UIntType;
  const T nan = __llvm_libc::fputil::FPBits<T>::buildNaN(1);
  const T inf = __llvm_libc::fputil::FPBits<T>::inf();
  const T negInf = __llvm_libc::fputil::FPBits<T>::negInf();
  const T zero = __llvm_libc::fputil::FPBits<T>::zero();
  const T negZero = __llvm_libc::fputil::FPBits<T>::negZero();

  UIntType getRandomBitPattern() {
    UIntType bits{0};
    for (UIntType i = 0; i < sizeof(UIntType) / 2; ++i) {
      bits =
          (bits << 2) + static_cast<uint16_t>(__llvm_libc::testutils::rand());
    }
    return bits;
  }

public:
  void testSpecialNumbers(Func func) {
    EXPECT_FP_EQ(func(zero, zero, zero), zero);
    EXPECT_FP_EQ(func(zero, negZero, negZero), negZero);
    EXPECT_FP_EQ(func(inf, inf, zero), inf);
    EXPECT_FP_EQ(func(negInf, inf, negInf), negInf);
    EXPECT_FP_EQ(func(inf, zero, zero), nan);
    EXPECT_FP_EQ(func(inf, negInf, inf), nan);
    EXPECT_FP_EQ(func(nan, zero, inf), nan);
    EXPECT_FP_EQ(func(inf, negInf, nan), nan);

    // Test underflow rounding up.
    EXPECT_FP_EQ(func(T(0.5), FPBits(FPBits::minSubnormal),
                      FPBits(FPBits::minSubnormal)),
                 FPBits(UIntType(2)));
    // Test underflow rounding down.
    FPBits v(FPBits::minNormal + UIntType(1));
    EXPECT_FP_EQ(
        func(T(1) / T(FPBits::minNormal << 1), v, FPBits(FPBits::minNormal)),
        v);
    // Test overflow.
    FPBits z(FPBits::maxNormal);
    EXPECT_FP_EQ(func(T(1.75), z, -z), T(0.75) * z);
  }

  void testSubnormalRange(Func func) {
    constexpr UIntType count = 1000001;
    constexpr UIntType step =
        (FPBits::maxSubnormal - FPBits::minSubnormal) / count;
    for (UIntType v = FPBits::minSubnormal, w = FPBits::maxSubnormal;
         v <= FPBits::maxSubnormal && w >= FPBits::minSubnormal;
         v += step, w -= step) {
      T x = FPBits(getRandomBitPattern()), y = FPBits(v), z = FPBits(w);
      T result = func(x, y, z);
      mpfr::TernaryInput<T> input{x, y, z};
      ASSERT_MPFR_MATCH(mpfr::Operation::Fma, input, result, 0.5);
    }
  }

  void testNormalRange(Func func) {
    constexpr UIntType count = 1000001;
    constexpr UIntType step = (FPBits::maxNormal - FPBits::minNormal) / count;
    for (UIntType v = FPBits::minNormal, w = FPBits::maxNormal;
         v <= FPBits::maxNormal && w >= FPBits::minNormal;
         v += step, w -= step) {
      T x = FPBits(v), y = FPBits(w), z = FPBits(getRandomBitPattern());
      T result = func(x, y, z);
      mpfr::TernaryInput<T> input{x, y, z};
      ASSERT_MPFR_MATCH(mpfr::Operation::Fma, input, result, 0.5);
    }
  }
};

#endif // LLVM_LIBC_TEST_SRC_MATH_FMATEST_H
