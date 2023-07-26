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
#include "src/stdio/fputs.h"
#include "src/stdio/fread.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcFOpenTest, PrintToFile) {
  int result;

  FILE *file = __llvm_libc::fopen("./testdata/test_data.txt", "w");
  ASSERT_FALSE(file == nullptr);

  static constexpr char STRING[] = "A simple string written to a file\n";
  result = __llvm_libc::fputs(STRING, file);
  EXPECT_GE(result, 0);

  ASSERT_EQ(0, __llvm_libc::fclose(file));

  FILE *new_file = __llvm_libc::fopen("./testdata/test_data.txt", "r");
  ASSERT_FALSE(new_file == nullptr);

  static char data[64] = {0};
  ASSERT_EQ(__llvm_libc::fread(data, 1, sizeof(STRING) - 1, new_file),
            sizeof(STRING) - 1);
  data[sizeof(STRING) - 1] = '\0';
  ASSERT_STREQ(data, STRING);

  ASSERT_EQ(0, __llvm_libc::fclose(new_file));
}
