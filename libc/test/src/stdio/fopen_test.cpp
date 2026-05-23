//===-- Unittests for fopen / fclose --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/scope.h"
#include "src/__support/File/file.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fwrite.h"

#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::scope_exit;

TEST(LlvmLibcFOpenTest, PrintToFile) {
  size_t result;

  static constexpr char STRING[] = "A simple string written to a file\n";
  {
    FILE *file =
        LIBC_NAMESPACE::fopen(APPEND_LIBC_TEST("testdata/test.txt"), "w");
    ASSERT_FALSE(file == nullptr);
    scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

    result = LIBC_NAMESPACE::fwrite(STRING, 1, sizeof(STRING) - 1, file);
    EXPECT_GE(result, size_t(0));
  }

  {
    FILE *file =
        LIBC_NAMESPACE::fopen(APPEND_LIBC_TEST("testdata/test.txt"), "r");
    ASSERT_FALSE(file == nullptr);
    scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

    static char data[64] = {0};
    ASSERT_EQ(LIBC_NAMESPACE::fread(data, 1, sizeof(STRING) - 1, file),
              sizeof(STRING) - 1);
    data[sizeof(STRING) - 1] = '\0';
    ASSERT_STREQ(data, STRING);
  }
}
