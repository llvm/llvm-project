//===-- Utility class to test fmul[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_FMULTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_FMULTEST_H

#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename OutType, typename InType>
class FmulMPFRTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(InType)

public:
  typedef OutType (*FMulFunc)(InType, InType);

  void testFMulMPFR(FMulFunc func) {
    constexpr int N = 10;
    mpfr::BinaryInput<InType> INPUTS[N] = {
        {3.0, 5.0},
        {0x1.0p1, 0x1.0p-131},
        {0x1.0p2, 0x1.0p-129},
        {1.0, 1.0},
        {-0.0, -0.0},
        {-0.0, 0.0},
        {0.0, -0.0},
        {0x1.0p100, 0x1.0p100},
        {1.0, 1.0 + 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150},
        {1.0, 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150}};

    for (int i = 0; i < N; ++i) {
      InType x = INPUTS[i].x;
      InType y = INPUTS[i].y;
      ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Fmul, INPUTS[i],
                                     func(x, y), 0.5);
    }
  }

  void testSpecialInputsMPFR(FMulFunc func) {
    constexpr int N = 27;
    mpfr::BinaryInput<InType> INPUTS[N] = {{inf, 0x1.0p-129},
                                           {0x1.0p-129, inf},
                                           {inf, 2.0},
                                           {3.0, inf},
                                           {0.0, 0.0},
                                           {neg_inf, aNaN},
                                           {aNaN, neg_inf},
                                           {neg_inf, neg_inf},
                                           {0.0, neg_inf},
                                           {neg_inf, 0.0},
                                           {neg_inf, 1.0},
                                           {1.0, neg_inf},
                                           {neg_inf, 0x1.0p-129},
                                           {0x1.0p-129, neg_inf},
                                           {0.0, 0x1.0p-129},
                                           {inf, 0.0},
                                           {0.0, inf},
                                           {0.0, aNaN},
                                           {2.0, aNaN},
                                           {0x1.0p-129, aNaN},
                                           {inf, aNaN},
                                           {aNaN, aNaN},
                                           {0.0, sNaN},
                                           {2.0, sNaN},
                                           {0x1.0p-129, sNaN},
                                           {inf, sNaN},
                                           {sNaN, sNaN}};

    for (int i = 0; i < N; ++i) {
      InType x = INPUTS[i].x;
      InType y = INPUTS[i].y;
      ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Fmul, INPUTS[i],
                                     func(x, y), 0.5);
    }
  }

  void testNormalRange(FMulFunc func) {
    using FPBits = LIBC_NAMESPACE::fputil::FPBits<InType>;
    using StorageType = typename FPBits::StorageType;
    static constexpr StorageType MAX_NORMAL = FPBits::max_normal().uintval();
    static constexpr StorageType MIN_NORMAL = FPBits::min_normal().uintval();

    constexpr StorageType COUNT = 10'001;
    constexpr StorageType STEP = (MAX_NORMAL - MIN_NORMAL) / COUNT;
    for (int signs = 0; signs < 4; ++signs) {
      for (StorageType v = MIN_NORMAL, w = MAX_NORMAL;
           v <= MAX_NORMAL && w >= MIN_NORMAL; v += STEP, w -= STEP) {
        InType x = FPBits(v).get_val(), y = FPBits(w).get_val();
        if (signs % 2 == 1) {
          x = -x;
        }
        if (signs >= 2) {
          y = -y;
        }

        mpfr::BinaryInput<InType> input{x, y};
        ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Fmul, input, func(x, y),
                                       0.5);
      }
    }
  }
};

#define LIST_FMUL_MPFR_TESTS(OutType, InType, func)                            \
  using LlvmLibcFmulTest = FmulMPFRTest<OutType, InType>;                      \
  TEST_F(LlvmLibcFmulTest, MulMpfr) { testFMulMPFR(&func); }                   \
  TEST_F(LlvmLibcFmulTest, NanInfMpfr) { testSpecialInputsMPFR(&func); }       \
  TEST_F(LlvmLibcFmulTest, NormalRange) { testNormalRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_FMULTEST_H
