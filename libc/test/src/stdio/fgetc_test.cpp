//===-- Unittests for fgetc -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/clearerr.h"
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fgetc.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"
#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <stdio.h>

TEST(LlvmLibcFGetCTest, WriteAndReadCharacters) {
  constexpr char FILENAME[] = "testdata/fgetc.test";
  ::FILE *file = __llvm_libc::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  constexpr char CONTENT[] = "123456789";
  constexpr size_t WRITE_SIZE = sizeof(CONTENT) - 1;
  ASSERT_EQ(WRITE_SIZE, __llvm_libc::fwrite(CONTENT, 1, WRITE_SIZE, file));
  // This is a write-only file so reads should fail.
  ASSERT_EQ(__llvm_libc::fgetc(file), EOF);
  // This is an error and not a real EOF.
  ASSERT_EQ(__llvm_libc::feof(file), 0);
  ASSERT_NE(__llvm_libc::ferror(file), 0);
  errno = 0;

  ASSERT_EQ(0, __llvm_libc::fclose(file));

  file = __llvm_libc::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  for (size_t i = 0; i < WRITE_SIZE; ++i) {
    int c = __llvm_libc::fgetc(file);
    ASSERT_EQ(c, int('1' + i));
  }
  // Reading more should return EOF but not set error.
  ASSERT_EQ(__llvm_libc::fgetc(file), EOF);
  ASSERT_NE(__llvm_libc::feof(file), 0);
  ASSERT_EQ(__llvm_libc::ferror(file), 0);

  ASSERT_EQ(0, __llvm_libc::fclose(file));
}
