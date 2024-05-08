//===-- Utility class to test FMod generic implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/ManipulationFunctions.h" // ldexp
#include "src/__support/FPUtil/generic/FMod.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <array>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename T, bool InverseMultiplication>
class LlvmLibcFModTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using U = typename FPBits::StorageType;
  using DivisionHelper = LIBC_NAMESPACE::cpp::conditional_t<
      InverseMultiplication,
      LIBC_NAMESPACE::fputil::generic::FModDivisionInvMultHelper<U>,
      LIBC_NAMESPACE::fputil::generic::FModDivisionSimpleHelper<U>>;

  static constexpr std::array<T, 11> TEST_BASES = {
      T(0.0),
      T(1.0),
      T(3.0),
      T(27.0),
      T(11.0 / 8.0),
      T(2.764443),
      T(1.0) - T(0x1.0p-23) - T(0x1.0p-52) - T(0x1.0p-112),
      T(1.0) + T(0x1.0p-23) + T(0x1.0p-52) + T(0x1.0p-112),
      T(3.14159265),
      T(1.41421356),
      T(2.71828183)};

public:
  void testExtensive() {
    using FMod = LIBC_NAMESPACE::fputil::generic::FMod<T, U, DivisionHelper>;
    int min2 = -(FPBits::MAX_BIASED_EXPONENT + FPBits::SIG_LEN) / 2;
    int max2 = 3 + FPBits::MAX_BIASED_EXPONENT / 2;
    for (T by : TEST_BASES) {
      for (int iy = min2; iy < max2; iy++) {
        T y = by * LIBC_NAMESPACE::fputil::ldexp(2.0, iy);
        FPBits y_bits(y);
        if (y_bits.is_zero() || !y_bits.is_finite())
          continue;
        for (T bx : TEST_BASES) {
          for (int ix = min2; ix < max2; ix++) {
            T x = bx * LIBC_NAMESPACE::fputil::ldexp(2.0, ix);
            if (!FPBits(x).is_finite())
              continue;
            T result = FMod::eval(x, y);
            mpfr::BinaryInput<T> input{x, y};
            EXPECT_MPFR_MATCH(mpfr::Operation::Fmod, input, result, 0.0);
          }
        }
      }
    }
  }
};

using LlvmLibcFModFloatTest = LlvmLibcFModTest<float, false>;
TEST_F(LlvmLibcFModFloatTest, ExtensiveTest) { testExtensive(); }

using LlvmLibcFModFloatInvTest = LlvmLibcFModTest<float, true>;
TEST_F(LlvmLibcFModFloatInvTest, ExtensiveTest) { testExtensive(); }

using LlvmLibcFModDoubleTest = LlvmLibcFModTest<double, false>;
TEST_F(LlvmLibcFModDoubleTest, ExtensiveTest) { testExtensive(); }

using LlvmLibcFModDoubleInvTest = LlvmLibcFModTest<double, true>;
TEST_F(LlvmLibcFModDoubleInvTest, ExtensiveTest) { testExtensive(); }
