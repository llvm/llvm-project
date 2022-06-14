//===-- Unittests for printf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf.h"

#include "utils/UnitTest/Test.h"

TEST(LlvmLibcPrintfTest, PrintOut) {
  int written;

  constexpr char simple[] = "A simple string with no conversions.\n";
  written = __llvm_libc::printf(simple);
  EXPECT_EQ(written, static_cast<int>(sizeof(simple) - 1));

  constexpr char numbers[] = "1234567890\n";
  written = __llvm_libc::printf("%s", numbers);
  EXPECT_EQ(written, static_cast<int>(sizeof(numbers) - 1));

  constexpr char format_more[] = "%s and more\n";
  constexpr char short_numbers[] = "1234";
  written = __llvm_libc::printf(format_more, short_numbers);
  EXPECT_EQ(written,
            static_cast<int>(sizeof(format_more) + sizeof(short_numbers) - 4));
}
