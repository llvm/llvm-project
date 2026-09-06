//===-- Utility class to test fixed-point sqrt ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/fixed_point/fx_rep.h"
#include "src/__support/fixed_point/sqrt.h"

template <typename ReturnType, typename FXType>
class SqrtTest : public LIBC_NAMESPACE::testing::Test {

  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<FXType>;
  using OutRep = LIBC_NAMESPACE::fixed_point::FXRep<ReturnType>;

  static constexpr ReturnType zero = OutRep::ZERO();
  static constexpr ReturnType half = OutRep::ONE_HALF();
  static constexpr ReturnType quarter = OutRep::ONE_FOURTH();
  static constexpr ReturnType eps = OutRep::EPS();
  static constexpr FXType in_max = FXRep::MAX();
  static constexpr FXType in_eps = FXRep::EPS();

public:
  typedef ReturnType (*SqrtFunc)(FXType);

  void testSpecialNumbers(SqrtFunc func) {
    constexpr double ERR = 3.0 * static_cast<double>(eps);
    auto check_error = [&](FXType v) {
      double v_d = static_cast<double>(v);
      double errors = LIBC_NAMESPACE::fputil::abs(
          static_cast<double>(func(v)) -
          LIBC_NAMESPACE::fputil::sqrt<double>(v_d));
      EXPECT_TRUE(errors <= ERR);
    };

    EXPECT_EQ(zero, func(zero));
    EXPECT_EQ(half, func(quarter));

    if constexpr (OutRep::INTEGRAL_LEN > 0) {
      EXPECT_EQ(static_cast<ReturnType>(1), func(1));
      EXPECT_EQ(static_cast<ReturnType>(2), func(4));
    }

    check_error(in_eps);

    using InputStorageType = typename FXRep::StorageType;

    constexpr size_t COUNT = 255;
    constexpr InputStorageType MAX_MAGNITUDE =
        InputStorageType(~InputStorageType(0));
    constexpr InputStorageType STEP =
        (MAX_MAGNITUDE < COUNT)
            ? 1
            : MAX_MAGNITUDE / static_cast<InputStorageType>(COUNT);

    InputStorageType x = 0;
    for (size_t i = 0; i < COUNT && x <= MAX_MAGNITUDE; ++i, x += STEP) {
      FXType v = LIBC_NAMESPACE::cpp::bit_cast<FXType>(x);
      check_error(v);
    }
    check_error(in_max);
  }
};

#define LIST_SQRT_TESTS(ReturnType, FXType, func)                              \
  using LlvmLibcSqrtTest = SqrtTest<ReturnType, FXType>;                       \
  TEST_F(LlvmLibcSqrtTest, SpecialNumbers) { testSpecialNumbers(&func); }      \
  static_assert(true, "Require semicolon.")
