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

#include "test/UnitTest/Test.h"

TEST(LlvmLibcFOpenTest, PrintToFile) {
  int result;

  FILE *file = __llvm_libc::fopen("./testdata/test_data.txt", "w");
  ASSERT_FALSE(file == nullptr);

  constexpr char another[] = "A simple string written to a file\n";
  result = __llvm_libc::fputs(another, file);
  EXPECT_GE(result, 0);

  ASSERT_EQ(0, __llvm_libc::fclose(file));
}
