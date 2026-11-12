//===-- Unittests for vprintf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These tests are copies of the non-v variants of the printf functions. This is
// because these functions are identical in every way except for how the varargs
// are passed.

#include "src/stdio/vprintf.h"

#include "test/UnitTest/Test.h"

int call_vprintf(const char *__restrict format, ...) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vprintf(format, vlist);
  va_end(vlist);
  return ret;
}

TEST(LlvmLibcVPrintfTest, PrintOut) {
  int written;

  constexpr char simple[] = "A simple string with no conversions.\n";
  written = call_vprintf(simple);
  EXPECT_EQ(written, static_cast<int>(sizeof(simple) - 1));

  constexpr char numbers[] = "1234567890\n";
  written = call_vprintf("%s", numbers);
  EXPECT_EQ(written, static_cast<int>(sizeof(numbers) - 1));

  constexpr char format_more[] = "%s and more\n";
  constexpr char short_numbers[] = "1234";
  written = call_vprintf(format_more, short_numbers);
  EXPECT_EQ(written,
            static_cast<int>(sizeof(format_more) + sizeof(short_numbers) - 4));
}
