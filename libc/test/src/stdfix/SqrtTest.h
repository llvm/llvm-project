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
  static constexpr ReturnType max = OutRep::MAX();
  static constexpr ReturnType half = OutRep::ONE_HALF();
  static constexpr ReturnType quarter = OutRep::ONE_FOURTH();
  static constexpr ReturnType one =
      (OutRep::INTEGRAL_LEN > 0) ? static_cast<ReturnType>(1) : OutRep::MAX();
  static constexpr ReturnType eps = OutRep::EPS();
  static constexpr FXType in_max = FXRep::MAX();
  static constexpr FXType in_eps = FXRep::EPS();

public:
  typedef ReturnType (*SqrtFunc)(FXType);

  void testSpecialNumbers(SqrtFunc func) {
    EXPECT_EQ(zero, func(zero));
    EXPECT_EQ(half, func(quarter));

    if constexpr (OutRep::INTEGRAL_LEN > 0) {
      EXPECT_EQ(one, func(one));
      EXPECT_EQ(static_cast<ReturnType>(2), func(4));
    }

    constexpr double ERR = 3.0 * static_cast<double>(eps);
    double eps_v_d = static_cast<double>(in_eps);
    double eps_error = LIBC_NAMESPACE::fputil::abs(
        static_cast<double>(func(in_eps)) -
        LIBC_NAMESPACE::fputil::sqrt<double>(eps_v_d));
    ASSERT_TRUE(eps_error <= ERR);

    using InputStorageType = typename FXRep::StorageType;

    constexpr size_t COUNT = 255;
    constexpr InputStorageType MAX_MAGNITUDE =
        (FXRep::SIGN_LEN > 0) ? static_cast<InputStorageType>(
                                    InputStorageType(~InputStorageType(0)) >> 1)
                              : InputStorageType(~InputStorageType(0));
    constexpr InputStorageType STEP =
        (MAX_MAGNITUDE < COUNT)
            ? 1
            : MAX_MAGNITUDE / static_cast<InputStorageType>(COUNT);

    InputStorageType x = 0;
    for (size_t i = 0; i < COUNT && x <= MAX_MAGNITUDE; ++i, x += STEP) {
      FXType v = LIBC_NAMESPACE::cpp::bit_cast<FXType>(x);
      double v_d = static_cast<double>(v);
      double errors = LIBC_NAMESPACE::fputil::abs(
          static_cast<double>(func(v)) -
          LIBC_NAMESPACE::fputil::sqrt<double>(v_d));
      if (errors > ERR) {
        // Print out the failure input and output.
        EXPECT_EQ(v, static_cast<FXType>(zero));
        EXPECT_EQ(func(v), zero);
      }
      ASSERT_TRUE(errors <= ERR);
    }
    double v_d = static_cast<double>(in_max);
    double error =
        LIBC_NAMESPACE::fputil::abs(static_cast<double>(func(in_max)) -
                                    LIBC_NAMESPACE::fputil::sqrt<double>(v_d));
    ASSERT_TRUE(error <= ERR);
  }
};

#define LIST_SQRT_TESTS(ReturnType, FXType, func)                              \
  using LlvmLibcSqrtTest = SqrtTest<ReturnType, FXType>;                       \
  TEST_F(LlvmLibcSqrtTest, SpecialNumbers) { testSpecialNumbers(&func); }      \
  static_assert(true, "Require semicolon.")
