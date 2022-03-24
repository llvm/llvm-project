//===-- Unittests for file operations like fopen, flcose etc --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fseek.h"
#include "src/stdio/fwrite.h"
#include "utils/UnitTest/Test.h"

#include <stdio.h>

TEST(LlvmLibcStdio, SimpleOperations) {
  constexpr char FILENAME[] = "testdata/simple_operations.test";
  ::FILE *file = __llvm_libc::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  constexpr char CONTENT[] = "1234567890987654321";
  ASSERT_EQ(sizeof(CONTENT) - 1,
            __llvm_libc::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file));
  ASSERT_EQ(0, __llvm_libc::fclose(file));

  file = __llvm_libc::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  constexpr size_t READ_SIZE = 5;
  char data[READ_SIZE];
  data[READ_SIZE - 1] = '\0';
  ASSERT_EQ(__llvm_libc::fread(data, 1, READ_SIZE - 1, file), READ_SIZE - 1);
  ASSERT_STREQ(data, "1234");
  ASSERT_EQ(__llvm_libc::fseek(file, 5, SEEK_CUR), 0);
  ASSERT_EQ(__llvm_libc::fread(data, 1, READ_SIZE - 1, file), READ_SIZE - 1);
  ASSERT_STREQ(data, "0987");
  ASSERT_EQ(__llvm_libc::fseek(file, -5, SEEK_CUR), 0);
  ASSERT_EQ(__llvm_libc::fread(data, 1, READ_SIZE - 1, file), READ_SIZE - 1);
  ASSERT_STREQ(data, "9098");

  ASSERT_EQ(__llvm_libc::fclose(file), 0);
}
