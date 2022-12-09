//===-- Unittests for fscanf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"

#include "src/stdio/fscanf.h"

#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <stdio.h>

TEST(LlvmLibcFScanfTest, WriteToFile) {
  constexpr char FILENAME[] = "testdata/fscanf_output.test";
  ::FILE *file = __llvm_libc::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  int read;

  constexpr char simple[] = "A simple string with no conversions.\n";

  ASSERT_EQ(sizeof(simple) - 1,
            __llvm_libc::fwrite(simple, 1, sizeof(simple) - 1, file));

  constexpr char numbers[] = "1234567890\n";

  ASSERT_EQ(sizeof(numbers) - 1,
            __llvm_libc::fwrite(numbers, 1, sizeof(numbers) - 1, file));

  constexpr char numbers_and_more[] = "1234 and more\n";

  ASSERT_EQ(sizeof(numbers_and_more) - 1,
            __llvm_libc::fwrite(numbers_and_more, 1,
                                sizeof(numbers_and_more) - 1, file));

  read =
      __llvm_libc::fscanf(file, "Reading from a write-only file should fail.");
  EXPECT_LT(read, 0);

  ASSERT_EQ(0, __llvm_libc::fclose(file));

  file = __llvm_libc::fopen(FILENAME, "r");
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

  ASSERT_EQ(__llvm_libc::ferror(file), 0);
  ASSERT_EQ(__llvm_libc::fclose(file), 0);
}
