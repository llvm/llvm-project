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

#include "src/regex/regerror.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/regex-macros.h"

TEST(LlvmLibcRegexTest, RegerrorAllCodes) {
  char buf[128];

  ASSERT_GT(LIBC_NAMESPACE::regerror(REG_NOMATCH, nullptr, buf, sizeof(buf)),
            size_t(0));
  ASSERT_STREQ("No match", buf);

  ASSERT_GT(LIBC_NAMESPACE::regerror(REG_ESPACE, nullptr, buf, sizeof(buf)),
            size_t(0));
  ASSERT_STREQ("Out of memory", buf);
}

TEST(LlvmLibcRegexTest, RegerrorTruncation) {
  char buf[5];
  size_t needed =
      LIBC_NAMESPACE::regerror(REG_NOMATCH, nullptr, buf, sizeof(buf));
  ASSERT_GT(needed, size_t(5)); // "No match" + NUL = 9 bytes
  ASSERT_EQ(buf[4], '\0');      // properly NUL-terminated
}

TEST(LlvmLibcRegexTest, RegerrorZeroBuffer) {
  size_t needed = LIBC_NAMESPACE::regerror(REG_NOMATCH, nullptr, nullptr, 0);
  ASSERT_GT(needed, size_t(0));
}

TEST(LlvmLibcRegexTest, RegerrorSuccess) {
  char buf[128];
  LIBC_NAMESPACE::regerror(0, nullptr, buf, sizeof(buf));
  ASSERT_STREQ("Success", buf);
}
