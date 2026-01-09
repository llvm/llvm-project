//===-- Unittests for wctype conversion utils -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/integer_literals.h"
#include "src/__support/wctype/conversion/utils/shared_utils.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace conversion_utils {

TEST(LlvmLibcMulHighTest, BasicCases) {
  EXPECT_EQ(mul_high(0_u64, 123_u64), 0_u64);
  EXPECT_EQ(mul_high(1_u64, 1_u64), 0_u64);
}

TEST(LlvmLibcMulHighTest, LargeValues) {
  uint64_t a = 0xFFFFFFFFFFFFFFFF;
  uint64_t b = 0xFFFFFFFFFFFFFFFF;

  uint64_t result = mul_high(a, b);

  // (2^64 - 1)^2 = 2^128 - 2^65 + 1
  EXPECT_EQ(result, 0xFFFFFFFFFFFFFFFE);
}

TEST(LlvmLibcWrappingMulTest, Basic) { EXPECT_EQ(wrapping_mul(5, 3), 15); }

TEST(LlvmLibcWrappingMulTest, Overflow) {
  EXPECT_EQ(wrapping_mul<uint8_t>(128_u8, 2_u8), 0_u8);
}

} // namespace conversion_utils

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL
