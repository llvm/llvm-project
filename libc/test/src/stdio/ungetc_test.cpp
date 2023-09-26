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
#include "test/UnitTest/Test.h"

#include <stdio.h>

TEST(LlvmLibcUngetcTest, UngetAndReadBack) {
  constexpr char FILENAME[] = "testdata/ungetc_test.test";
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  constexpr char CONTENT[] = "abcdef";
  constexpr size_t CONTENT_SIZE = sizeof(CONTENT);
  ASSERT_EQ(CONTENT_SIZE,
            LIBC_NAMESPACE::fwrite(CONTENT, 1, CONTENT_SIZE, file));
  // Cannot unget to an un-readable file.
  ASSERT_EQ(EOF, LIBC_NAMESPACE::ungetc('1', file));
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));

  file = LIBC_NAMESPACE::fopen(FILENAME, "r+");
  ASSERT_FALSE(file == nullptr);
  char c;
  ASSERT_EQ(LIBC_NAMESPACE::fread(&c, 1, 1, file), size_t(1));
  ASSERT_EQ(c, CONTENT[0]);
  ASSERT_EQ(LIBC_NAMESPACE::ungetc(int(c), file), int(c));

  char data[CONTENT_SIZE];
  ASSERT_EQ(CONTENT_SIZE, LIBC_NAMESPACE::fread(data, 1, CONTENT_SIZE, file));
  ASSERT_STREQ(CONTENT, data);

  ASSERT_EQ(0, LIBC_NAMESPACE::fseek(file, 0, SEEK_SET));
  // ungetc should not fail after a seek operation.
  int unget_char = 'z';
  ASSERT_EQ(unget_char, LIBC_NAMESPACE::ungetc(unget_char, file));
  // Another unget should fail.
  ASSERT_EQ(EOF, LIBC_NAMESPACE::ungetc(unget_char, file));
  // ungetting a char at the beginning of the file will allow us to fetch
  // one additional character.
  char new_data[CONTENT_SIZE + 1];
  ASSERT_EQ(CONTENT_SIZE + 1,
            LIBC_NAMESPACE::fread(new_data, 1, CONTENT_SIZE + 1, file));
  ASSERT_STREQ("zabcdef", new_data);

  ASSERT_EQ(size_t(1), LIBC_NAMESPACE::fwrite("x", 1, 1, file));
  // unget should fail after a write operation.
  ASSERT_EQ(EOF, LIBC_NAMESPACE::ungetc('1', file));

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));
}
