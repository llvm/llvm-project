//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains smoke tests for static_rounding::expf(x)
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/expf.h"
#include "src/__support/math/expf_integer_eval.h"
#include "src/math/expf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExpfTest = LIBC_NAMESPACE::testing::FPTest<float>;

// TODO: add tests

// TEST_F(LlvmLibcExpfTest, SpecialNumbers) {
//   EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::expf(sNaN));
//   EXPECT_MATH_ERRNO(0);

//   EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::expf(aNaN));
//   EXPECT_MATH_ERRNO(0);

//   EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::expf(inf));
//   EXPECT_MATH_ERRNO(0);

//   EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::expf(neg_inf));
//   EXPECT_MATH_ERRNO(0);

//   EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::expf(0.0f));
//   EXPECT_MATH_ERRNO(0);

//   EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::expf(-0.0f));
//   EXPECT_MATH_ERRNO(0);
// }

// TEST_F(LlvmLibcExpfTest, Overflow) {
//   using LIBC_NAMESPACE::shared::math::static_rounding::expf;
//   EXPECT_FP_EQ_ALL_ROUNDING(
//       inf, LIBC_NAMESPACE::expf(FPBits(0x7f7fffffU).get_val()));
//   EXPECT_MATH_ERRNO(0);

//   EXPECT_FP_EQ_ALL_ROUNDING(
//       inf, LIBC_NAMESPACE::expf(FPBits(0x42cffff8U).get_val()));
//   EXPECT_MATH_ERRNO(0);

//   EXPECT_FP_EQ_ALL_ROUNDING(
//       inf, LIBC_NAMESPACE::expf(FPBits(0x42d00008U).get_val()));
//   EXPECT_MATH_ERRNO(0);

//   constexpr float X = FPBits(0xC2CF'F1B2U).get_val();
//   EXPECT_FP_EQ_ROUNDING_NEAREST(
//     LIBC_NAMESPACE::expf(X), expf(X, FE_TONEAREST)
//   );
//   EXPECT_MATH_ERRNO(0);
// }

TEST_F(LlvmLibcExpfTest, Special) {
  float x = FPBits(3266354647U).get_val();
  EXPECT_FP_EQ_ROUNDING_NEAREST(
      LIBC_NAMESPACE::math::expf(x),
      LIBC_NAMESPACE::shared::math::static_rounding::expf(x, FE_TONEAREST));
  EXPECT_MATH_ERRNO(0);
}
