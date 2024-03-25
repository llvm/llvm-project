//===-- Unittests for fopen / fclose --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/fread.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcFOpenTest, PrintToFile) {
  int result;

  FILE *file = LIBC_NAMESPACE::fopen("./testdata/test_data.txt", "w");
  ASSERT_FALSE(file == nullptr);

  static constexpr char STRING[] = "A simple string written to a file\n";
  result = LIBC_NAMESPACE::fwrite(STRING, 1, sizeof(STRING) - 1, file);
  EXPECT_GE(result, 0);

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));

  FILE *new_file = LIBC_NAMESPACE::fopen("./testdata/test_data.txt", "r");
  ASSERT_FALSE(new_file == nullptr);

  static char data[64] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(data, 1, sizeof(STRING) - 1, new_file),
            sizeof(STRING) - 1);
  data[sizeof(STRING) - 1] = '\0';
  ASSERT_STREQ(data, STRING);

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(new_file));
}
