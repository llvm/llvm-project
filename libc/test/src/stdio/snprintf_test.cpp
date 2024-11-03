//===-- Unittests for snprintf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/snprintf.h"

#include "test/UnitTest/Test.h"

// The sprintf test cases cover testing the shared printf functionality, so
// these tests will focus on snprintf exclusive features.

TEST(LlvmLibcSNPrintfTest, CutOff) {
  char buff[100];
  int written;

  written =
      __llvm_libc::snprintf(buff, 16, "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string");

  written = __llvm_libc::snprintf(buff, 5, "%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "1234");

  written = __llvm_libc::snprintf(buff, 67, "%-101c", 'a');
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

  // passing null as the output pointer is allowed as long as buffsz is 0.
  written = __llvm_libc::snprintf(nullptr, 0, "%s and more", "1234567890");
  EXPECT_EQ(written, 19);
}

TEST(LlvmLibcSNPrintfTest, NoCutOff) {
  char buff[64];
  int written;

  written =
      __llvm_libc::snprintf(buff, 37, "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string with no conversions.");

  written = __llvm_libc::snprintf(buff, 20, "%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "1234567890");
}
