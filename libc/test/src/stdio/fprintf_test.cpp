//===-- Unittests for fprintf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_COPT_PRINTF_USE_SYSTEM_FILE
#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#endif // LIBC_COPT_PRINTF_USE_SYSTEM_FILE

#include "src/stdio/fprintf.h"

#include "test/UnitTest/Test.h"

#include <stdio.h>

namespace printf_test {
#ifndef LIBC_COPT_PRINTF_USE_SYSTEM_FILE
using __llvm_libc::fclose;
using __llvm_libc::ferror;
using __llvm_libc::fopen;
using __llvm_libc::fread;
#else  // defined(LIBC_COPT_PRINTF_USE_SYSTEM_FILE)
using ::fclose;
using ::ferror;
using ::fopen;
using ::fread;
#endif // LIBC_COPT_PRINTF_USE_SYSTEM_FILE
} // namespace printf_test

TEST(LlvmLibcFPrintfTest, WriteToFile) {
  const char *FILENAME = "fprintf_output.test";
  auto FILE_PATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = printf_test::fopen(FILE_PATH, "w");
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

  ASSERT_EQ(0, printf_test::fclose(file));

  file = printf_test::fopen(FILE_PATH, "r");
  ASSERT_FALSE(file == nullptr);

  char data[50];
  ASSERT_EQ(printf_test::fread(data, 1, sizeof(simple) - 1, file),
            sizeof(simple) - 1);
  data[sizeof(simple) - 1] = '\0';
  ASSERT_STREQ(data, simple);
  ASSERT_EQ(printf_test::fread(data, 1, sizeof(numbers) - 1, file),
            sizeof(numbers) - 1);
  data[sizeof(numbers) - 1] = '\0';
  ASSERT_STREQ(data, numbers);
  ASSERT_EQ(printf_test::fread(
                data, 1, sizeof(format_more) + sizeof(short_numbers) - 4, file),
            sizeof(format_more) + sizeof(short_numbers) - 4);
  data[sizeof(format_more) + sizeof(short_numbers) - 4] = '\0';
  ASSERT_STREQ(data, "1234 and more\n");

  ASSERT_EQ(printf_test::ferror(file), 0);

  written =
      __llvm_libc::fprintf(file, "Writing to a read only file should fail.");
  EXPECT_LT(written, 0);

  ASSERT_EQ(printf_test::fclose(file), 0);
}
