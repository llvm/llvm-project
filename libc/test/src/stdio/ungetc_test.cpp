//===-- Unittests for ungetc ----------------------------------------------===//
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
#include "src/stdio/ungetc.h"
#include "utils/UnitTest/Test.h"

#include <stdio.h>

TEST(LlvmLibcUngetcTest, UngetAndReadBack) {
  constexpr char FILENAME[] = "testdata/ungetc_test.test";
  ::FILE *file = __llvm_libc::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  constexpr char CONTENT[] = "abcdef";
  constexpr size_t CONTENT_SIZE = sizeof(CONTENT);
  ASSERT_EQ(CONTENT_SIZE, __llvm_libc::fwrite(CONTENT, 1, CONTENT_SIZE, file));
  // Cannot unget to an un-readable file.
  ASSERT_EQ(EOF, __llvm_libc::ungetc('1', file));
  ASSERT_EQ(0, __llvm_libc::fclose(file));

  file = __llvm_libc::fopen(FILENAME, "r+");
  ASSERT_FALSE(file == nullptr);
  char c;
  ASSERT_EQ(__llvm_libc::fread(&c, 1, 1, file), size_t(1));
  ASSERT_EQ(c, CONTENT[0]);
  ASSERT_EQ(__llvm_libc::ungetc(int(c), file), int(c));

  char data[CONTENT_SIZE];
  ASSERT_EQ(CONTENT_SIZE, __llvm_libc::fread(data, 1, CONTENT_SIZE, file));
  ASSERT_STREQ(CONTENT, data);

  ASSERT_EQ(0, __llvm_libc::fseek(file, 0, SEEK_SET));
  // ungetc should not fail after a seek operation.
  int unget_char = 'z';
  ASSERT_EQ(unget_char, __llvm_libc::ungetc(unget_char, file));
  // Another unget should fail.
  ASSERT_EQ(EOF, __llvm_libc::ungetc(unget_char, file));
  // ungetting a char at the beginning of the file will allow us to fetch
  // one additional character.
  char new_data[CONTENT_SIZE + 1];
  ASSERT_EQ(CONTENT_SIZE + 1,
            __llvm_libc::fread(new_data, 1, CONTENT_SIZE + 1, file));
  ASSERT_STREQ("zabcdef", new_data);

  ASSERT_EQ(size_t(1), __llvm_libc::fwrite("x", 1, 1, file));
  // unget should fail after a write operation.
  ASSERT_EQ(EOF, __llvm_libc::ungetc('1', file));

  ASSERT_EQ(0, __llvm_libc::fclose(file));
}
