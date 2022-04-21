//===-- Unittests for file operations like fopen, flcose etc --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/flockfile.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread_unlocked.h"
#include "src/stdio/funlockfile.h"
#include "src/stdio/fwrite_unlocked.h"
#include "utils/UnitTest/Test.h"

#include <stdio.h>

TEST(LlvmLibcFILETest, UnlockedReadAndWrite) {
  constexpr char FILENAME[] = "testdata/unlocked_read_and_write.test";
  ::FILE *file = __llvm_libc::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  constexpr char CONTENT[] = "1234567890987654321";
  __llvm_libc::flockfile(file);
  ASSERT_EQ(sizeof(CONTENT) - 1, __llvm_libc::fwrite_unlocked(
                                     CONTENT, 1, sizeof(CONTENT) - 1, file));
  __llvm_libc::funlockfile(file);
  ASSERT_EQ(0, __llvm_libc::fclose(file));

  file = __llvm_libc::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  constexpr size_t READ_SIZE = 5;
  char data[READ_SIZE * 2 + 1];
  data[READ_SIZE * 2] = '\0';
  __llvm_libc::flockfile(file);
  ASSERT_EQ(__llvm_libc::fread_unlocked(data, 1, READ_SIZE, file), READ_SIZE);
  ASSERT_EQ(__llvm_libc::fread_unlocked(data + READ_SIZE, 1, READ_SIZE, file),
            READ_SIZE);
  __llvm_libc::funlockfile(file);
  ASSERT_STREQ(data, "1234567890");

  ASSERT_EQ(__llvm_libc::fclose(file), 0);
}
