//===-- Utility class to test FMod generic implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/generic/FMod.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"

#include <array>
#include <limits>

namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T, bool InverseMultiplication>
class LlvmLibcFModTest : public __llvm_libc::testing::Test {

  using DivisionHelper = __llvm_libc::cpp::conditional_t<
      InverseMultiplication,
      __llvm_libc::fputil::generic::FModDivisionInvMultHelper<T>,
      __llvm_libc::fputil::generic::FModDivisionSimpleHelper<T>>;

  static constexpr std::array<T, 11> test_bases = {
      T(0.0),
      T(1.0),
      T(3.0),
      T(27.0),
      T(11.0 / 8.0),
      T(2.764443),
      T(1.0) - std::numeric_limits<T>::epsilon(),
      T(1.0) + std::numeric_limits<T>::epsilon(),
      T(M_PI),
      T(M_SQRT2),
      T(M_E)};

public:
  void testExtensive() {
    using FMod = __llvm_libc::fputil::generic::FMod<
        T, __llvm_libc::fputil::generic::FModFastMathWrapper<T>,
        DivisionHelper>;
    using nl = std::numeric_limits<T>;
    int min2 = nl::min_exponent - nl::digits - 5;
    int max2 = nl::max_exponent + 3;
    for (T by : test_bases) {
      for (int iy = min2; iy < max2; iy++) {
        T y = by * std::ldexp(2, iy);
        if (y == 0 || !std::isfinite(y))
          continue;
        for (T bx : test_bases) {
          for (int ix = min2; ix < max2; ix++) {
            T x = bx * std::ldexp(2, ix);
            if (!std::isfinite(x))
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
