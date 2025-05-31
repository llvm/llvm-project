//===-- Unittests for asprintf--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/asprintf.h"
#include "src/stdio/sprintf.h"
#include "src/string/memset.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcASPrintfTest, SimpleNoConv) {
  char *buff = nullptr;
  int written;
  written =
      LIBC_NAMESPACE::asprintf(&buff, "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string with no conversions.");
  free(buff);
}

TEST(LlvmLibcASPrintfTest, PercentConv) {
  char *buff = nullptr;
  int written;

  written = LIBC_NAMESPACE::asprintf(&buff, "%%");
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "%");
  free(buff);

  written = LIBC_NAMESPACE::asprintf(&buff, "abc %% def");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "abc % def");
  free(buff);

  written = LIBC_NAMESPACE::asprintf(&buff, "%%%%%%");
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "%%%");
  free(buff);
}

TEST(LlvmLibcASPrintfTest, CharConv) {
  char *buff = nullptr;
  int written;

  written = LIBC_NAMESPACE::asprintf(&buff, "%c", 'a');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "a");
  free(buff);

  written = LIBC_NAMESPACE::asprintf(&buff, "%3c %-3c", '1', '2');
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  1 2  ");
  free(buff);

  written = LIBC_NAMESPACE::asprintf(&buff, "%*c", 2, '3');
  EXPECT_EQ(written, 2);
  ASSERT_STREQ(buff, " 3");
  free(buff);
}

TEST(LlvmLibcASPrintfTest, LargeStringNoConv) {
  char *buff = nullptr;
  char long_str[1001];
  LIBC_NAMESPACE::memset(long_str, 'a', 1000);
  long_str[1000] = '\0';
  int written;
  written = LIBC_NAMESPACE::asprintf(&buff, long_str);
  EXPECT_EQ(written, 1000);
  ASSERT_STREQ(buff, long_str);
  free(buff);
}

TEST(LlvmLibcASPrintfTest, ManyReAlloc) {
  char *buff = nullptr;
  char long_str[1001];
  auto expected_num_chars =
      LIBC_NAMESPACE::sprintf(long_str, "%200s%200s%200s", "a", "b", "c");
  long_str[expected_num_chars] = '\0';
  int written;
  written = LIBC_NAMESPACE::asprintf(&buff, long_str);
  EXPECT_EQ(written, expected_num_chars);
  ASSERT_STREQ(buff, long_str);
  free(buff);
}
