//===-- Unittests for snprintf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These tests are copies of the non-v variants of the printf functions. This is
// because these functions are identical in every way except for how the varargs
// are passed.

#include "src/stdio/vsnprintf.h"

#include "test/UnitTest/Test.h"

int call_vsnprintf(char *__restrict buffer, size_t buffsz,
                   const char *__restrict format, ...) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vsnprintf(buffer, buffsz, format, vlist);
  va_end(vlist);
  return ret;
}

// The sprintf test cases cover testing the shared printf functionality, so
// these tests will focus on snprintf exclusive features.

TEST(LlvmLibcVSNPrintfTest, CutOff) {
  char buff[100];
  int written;

  written = call_vsnprintf(buff, 16, "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string");

  written = call_vsnprintf(buff, 5, "%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "1234");

  written = call_vsnprintf(buff, 67, "%-101c", 'a');
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
  written = call_vsnprintf(nullptr, 0, "%s and more", "1234567890");
  EXPECT_EQ(written, 19);
}

TEST(LlvmLibcVSNPrintfTest, NoCutOff) {
  char buff[64];
  int written;

  written = call_vsnprintf(buff, 37, "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string with no conversions.");

  written = call_vsnprintf(buff, 20, "%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "1234567890");
}
