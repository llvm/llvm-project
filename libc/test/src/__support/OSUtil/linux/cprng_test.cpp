//===-- Unittests for CPRNG -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/OSUtil/linux/cprng.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {
namespace cprng {
TEST(LlvmLibcOSUtilCPRNGTest, Generate) {
  auto result = generate<uint32_t>();
  ASSERT_TRUE(result.has_value());
}
TEST(LlvmLibcOSUtilCPRNGTest, GenerateBounded) {
  for (uint32_t bound = 1; bound < 5000; ++bound) {
    auto result = generate_bounded_u32(bound);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(*result, bound);
  }
}
} // namespace cprng
} // namespace LIBC_NAMESPACE_DECL
