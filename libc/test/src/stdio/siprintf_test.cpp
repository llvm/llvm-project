//===-- Unittests for siprintf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/siprintf.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcSIPrintfTest, SimpleParsing) {
  char buff[100];
  int written;

  constexpr char simple[] = "A simple string with no conversions.";
  written = LIBC_NAMESPACE::siprintf(buff, simple);
  EXPECT_EQ(written, static_cast<int>(sizeof(simple) - 1));
  ASSERT_STREQ(buff, simple);

  constexpr char numbers[] = "1234567890";
  written = LIBC_NAMESPACE::siprintf(buff, "%s", numbers);
  EXPECT_EQ(written, static_cast<int>(sizeof(numbers) - 1));
  ASSERT_STREQ(buff, numbers);

  constexpr char format_more[] = "%s and more";
  constexpr char short_numbers[] = "1234";
  written = LIBC_NAMESPACE::siprintf(buff, format_more, short_numbers);
  EXPECT_EQ(written,
            static_cast<int>(sizeof(format_more) + sizeof(short_numbers) - 4));
  ASSERT_STREQ(buff, "1234 and more");

  constexpr char format_float[] = "%f doesn't work\n";
  written = LIBC_NAMESPACE::siprintf(buff, format_float, 1.0);
  EXPECT_EQ(written, static_cast<int>(sizeof(format_float) - 1));
  ASSERT_STREQ(buff, format_float);
}
