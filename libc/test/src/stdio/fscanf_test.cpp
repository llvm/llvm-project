//===-- Unittests for fscanf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"

#ifndef LIBC_COPT_STDIO_USE_SYSTEM_FILE
#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"
#endif // LIBC_COPT_STDIO_USE_SYSTEM_FILE

#include "src/stdio/fscanf.h"

#include "test/UnitTest/Test.h"

#include <stdio.h>

namespace scanf_test {
#ifndef LIBC_COPT_STDIO_USE_SYSTEM_FILE
using __llvm_libc::fclose;
using __llvm_libc::ferror;
using __llvm_libc::fopen;
using __llvm_libc::fwrite;
#else  // defined(LIBC_COPT_STDIO_USE_SYSTEM_FILE)
using ::fclose;
using ::ferror;
using ::fopen;
using ::fwrite;
#endif // LIBC_COPT_STDIO_USE_SYSTEM_FILE
} // namespace scanf_test

TEST(LlvmLibcFScanfTest, WriteToFile) {
  const char *FILENAME = "fscanf_output.test";
  auto FILE_PATH = libc_make_test_file_path(FILENAME);
  ::FILE *file = scanf_test::fopen(FILE_PATH, "w");
  ASSERT_FALSE(file == nullptr);

  int read;

  constexpr char simple[] = "A simple string with no conversions.\n";

  ASSERT_EQ(sizeof(simple) - 1,
            scanf_test::fwrite(simple, 1, sizeof(simple) - 1, file));

  constexpr char numbers[] = "1234567890\n";

  ASSERT_EQ(sizeof(numbers) - 1,
            scanf_test::fwrite(numbers, 1, sizeof(numbers) - 1, file));

  constexpr char numbers_and_more[] = "1234 and more\n";

  ASSERT_EQ(sizeof(numbers_and_more) - 1,
            scanf_test::fwrite(numbers_and_more, 1,
                               sizeof(numbers_and_more) - 1, file));

  read =
      __llvm_libc::fscanf(file, "Reading from a write-only file should fail.");
  EXPECT_LT(read, 0);

  ASSERT_EQ(0, scanf_test::fclose(file));

  file = scanf_test::fopen(FILE_PATH, "r");
  ASSERT_FALSE(file == nullptr);

  char data[50];
  read = __llvm_libc::fscanf(file, "%[A-Za-z .\n]", data);
  ASSERT_EQ(read, 1);
  ASSERT_STREQ(simple, data);

  read = __llvm_libc::fscanf(file, "%s", data);
  ASSERT_EQ(read, 1);
  ASSERT_EQ(__llvm_libc::cpp::string_view(numbers, 10),
            __llvm_libc::cpp::string_view(data));

  // The format string starts with a space to handle the fact that the %s leaves
  // a trailing \n and %c doesn't strip leading whitespace.
  read = __llvm_libc::fscanf(file, " %50c", data);
  ASSERT_EQ(read, 1);
  ASSERT_EQ(__llvm_libc::cpp::string_view(numbers_and_more),
            __llvm_libc::cpp::string_view(data, sizeof(numbers_and_more) - 1));

  ASSERT_EQ(scanf_test::ferror(file), 0);
  ASSERT_EQ(scanf_test::fclose(file), 0);
}
