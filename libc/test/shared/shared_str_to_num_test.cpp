//===-- Unittests for shared string to number functions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shared/str_to_float.h"
#include "shared/str_to_integer.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcSharedStrToNumTest, IntegerTests) {
  {
    auto result = shared::strtointeger<int>("123", 10);
    EXPECT_EQ(result.value, 123);
    EXPECT_EQ(result.parsed_len, ptrdiff_t(3));
    EXPECT_EQ(result.error, 0);
  }
  {
    auto result = shared::strtointeger<int>("  -0x123", 0);
    EXPECT_EQ(result.value, -0x123);
    EXPECT_EQ(result.parsed_len, ptrdiff_t(8));
    EXPECT_EQ(result.error, 0);
  }
}

TEST(LlvmLibcSharedStrToNumTest, FloatTests) {
  {
    // 1.25 = 1.01b = 5 * 2^-2
    shared::ExpandedFloat<double> input;
    input.mantissa = 5;
    input.exponent = -2;
    auto result = shared::binary_exp_to_float<double>(
        input, false, shared::RoundDirection::Nearest);
    EXPECT_EQ(result.num.mantissa, uint64_t(0x4000000000000));
    EXPECT_EQ(result.num.exponent, 1023);
    EXPECT_EQ(result.error, 0);
  }
  {
    // 1.25 = 125 * 10^-2
    shared::ExpandedFloat<double> input;
    input.mantissa = 125;
    input.exponent = -2;
    const char *str = "1.25";
    auto result = shared::decimal_exp_to_float<double>(
        input, false, shared::RoundDirection::Nearest, str);
    EXPECT_EQ(result.num.mantissa, uint64_t(0x14000000000000));
    EXPECT_EQ(result.num.exponent, 1023);
    EXPECT_EQ(result.error, 0);
  }
}

} // namespace LIBC_NAMESPACE_DECL
