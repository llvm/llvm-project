//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for regerror.
///
//===----------------------------------------------------------------------===//

#include "hdr/regex_macros.h"
#include "src/regex/regerror.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcRegexTest, RegerrorBasicCodes) {
  char buf[128];

  EXPECT_EQ(LIBC_NAMESPACE::regerror(REG_NOMATCH, nullptr, buf, sizeof(buf)),
            size_t(9));
  EXPECT_STREQ(buf, "No match");

  EXPECT_EQ(LIBC_NAMESPACE::regerror(REG_ESPACE, nullptr, buf, sizeof(buf)),
            size_t(14));
  EXPECT_STREQ(buf, "Out of memory");

  EXPECT_EQ(LIBC_NAMESPACE::regerror(-1, nullptr, buf, sizeof(buf)),
            size_t(14));
  EXPECT_STREQ(buf, "Unknown error");
}

TEST(LlvmLibcRegexTest, RegerrorTruncation) {
  char buf[5];
  size_t needed =
      LIBC_NAMESPACE::regerror(REG_NOMATCH, nullptr, buf, sizeof(buf));
  EXPECT_EQ(needed, size_t(9)); // strlen("No match") + 1
  EXPECT_STREQ(buf, "No m"); // "No match" truncated to 5 bytes (including NUL)

  char single[1] = {'x'};
  EXPECT_EQ(
      LIBC_NAMESPACE::regerror(REG_NOMATCH, nullptr, single, sizeof(single)),
      size_t(9));
  EXPECT_STREQ(single, "");
}

TEST(LlvmLibcRegexTest, RegerrorZeroBuffer) {
  size_t needed = LIBC_NAMESPACE::regerror(REG_NOMATCH, nullptr, nullptr, 0);
  EXPECT_EQ(needed, size_t(9));
}

TEST(LlvmLibcRegexTest, RegerrorSuccess) {
  char buf[128];
  EXPECT_EQ(LIBC_NAMESPACE::regerror(0, nullptr, buf, sizeof(buf)), size_t(8));
  EXPECT_STREQ(buf, "Success");
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST(LlvmLibcRegexTest, RegerrorNullPointerCrash) {
  ASSERT_DEATH(
      [] { LIBC_NAMESPACE::regerror(REG_NOMATCH, nullptr, nullptr, 1); },
      WITH_SIGNAL(-1));
}
#endif // defined(LIBC_ADD_NULL_CHECKS)
