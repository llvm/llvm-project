//===-- Unittests for sniprintf -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/sniprintf.h"

#include "test/UnitTest/Test.h"

// The sprintf test cases cover testing the shared printf functionality, so
// these tests will focus on sniprintf exclusive features.

TEST(LlvmLibcSNIPrintfTest, CutOff) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::sniprintf(buff, 16,
                                      "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string");

  written = LIBC_NAMESPACE::sniprintf(buff, 5, "%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "1234");

  written = LIBC_NAMESPACE::sniprintf(buff, 67, "%-101c", 'a');
  EXPECT_EQ(written, 101);
  ASSERT_STREQ(buff, "a "
                     "        " // Each of these is 8 spaces, and there are 8.
                     "        " // In total there are 65 spaces
                     "        " // 'a' + 65 spaces + '\0' = 67
                     "        "
                     "        "
                     "        "
                     "        "
                     "        ");

  written =
      LIBC_NAMESPACE::sniprintf(buff, 10, "%f before and %f after", 1.0, 2.0);
  EXPECT_EQ(written, 22);
  ASSERT_STREQ(buff, "%f before");

  // passing null as the output pointer is allowed as long as buffsz is 0.
  written = LIBC_NAMESPACE::sniprintf(nullptr, 0, "%s and more", "1234567890");
  EXPECT_EQ(written, 19);

  written = LIBC_NAMESPACE::sniprintf(nullptr, 0, "%*s", INT_MIN, "nothing");
  EXPECT_EQ(written, INT_MAX);
}

TEST(LlvmLibcSNIPrintfTest, NoCutOff) {
  char buff[64];
  int written;

  written = LIBC_NAMESPACE::sniprintf(buff, 37,
                                      "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string with no conversions.");

  written = LIBC_NAMESPACE::sniprintf(buff, 20, "%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "1234567890");
}
