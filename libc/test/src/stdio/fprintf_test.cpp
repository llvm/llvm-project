//===-- Unittests for fprintf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"

#include "src/stdio/fprintf.h"

#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <stdio.h>

TEST(LlvmLibcFPrintfTest, WriteToFile) {
  constexpr char FILENAME[] = "testdata/fprintf_output.test";
  ::FILE *file = __llvm_libc::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  int written;

  constexpr char simple[] = "A simple string with no conversions.\n";
  written = __llvm_libc::fprintf(file, simple);
  EXPECT_EQ(written, 37);

  constexpr char numbers[] = "1234567890\n";
  written = __llvm_libc::fprintf(file, "%s", numbers);
  EXPECT_EQ(written, 11);

  constexpr char format_more[] = "%s and more\n";
  constexpr char short_numbers[] = "1234";
  written = __llvm_libc::fprintf(file, format_more, short_numbers);
  EXPECT_EQ(written, 14);

  ASSERT_EQ(0, __llvm_libc::fclose(file));

  file = __llvm_libc::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  char data[50];
  ASSERT_EQ(__llvm_libc::fread(data, 1, sizeof(simple) - 1, file),
            sizeof(simple) - 1);
  data[sizeof(simple) - 1] = '\0';
  ASSERT_STREQ(data, simple);
  ASSERT_EQ(__llvm_libc::fread(data, 1, sizeof(numbers) - 1, file),
            sizeof(numbers) - 1);
  data[sizeof(numbers) - 1] = '\0';
  ASSERT_STREQ(data, numbers);
  ASSERT_EQ(__llvm_libc::fread(
                data, 1, sizeof(format_more) + sizeof(short_numbers) - 4, file),
            sizeof(format_more) + sizeof(short_numbers) - 4);
  data[sizeof(format_more) + sizeof(short_numbers) - 4] = '\0';
  ASSERT_STREQ(data, "1234 and more\n");

  ASSERT_EQ(__llvm_libc::ferror(file), 0);

  written =
      __llvm_libc::fprintf(file, "Writing to a read only file should fail.");
  EXPECT_LT(written, 0);

  ASSERT_EQ(__llvm_libc::fclose(file), 0);
}
