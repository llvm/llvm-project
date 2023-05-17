//===-- Unittests for setbuf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/setbuf.h"
#include "src/stdio/ungetc.h"
#include "test/UnitTest/Test.h"

#include <stdio.h>

TEST(LlvmLibcSetbufTest, DefaultBufsize) {
  // The idea in this test is to change the buffer after opening a file and
  // ensure that read and write work as expected.
  constexpr char FILENAME[] = "testdata/setbuf_test_default_bufsize.test";
  ::FILE *file = __llvm_libc::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  char buffer[BUFSIZ];
  __llvm_libc::setbuf(file, buffer);
  constexpr char CONTENT[] = "abcdef";
  constexpr size_t CONTENT_SIZE = sizeof(CONTENT);
  ASSERT_EQ(CONTENT_SIZE, __llvm_libc::fwrite(CONTENT, 1, CONTENT_SIZE, file));
  ASSERT_EQ(0, __llvm_libc::fclose(file));

  file = __llvm_libc::fopen(FILENAME, "r");
  __llvm_libc::setbuf(file, buffer);
  ASSERT_FALSE(file == nullptr);
  char data[CONTENT_SIZE];
  ASSERT_EQ(__llvm_libc::fread(&data, 1, CONTENT_SIZE, file), CONTENT_SIZE);
  ASSERT_STREQ(CONTENT, data);
  ASSERT_EQ(0, __llvm_libc::fclose(file));
}

TEST(LlvmLibcSetbufTest, NullBuffer) {
  // The idea in this test is that we set a null buffer and ensure that
  // everything works correctly.
  constexpr char FILENAME[] = "testdata/setbuf_test_null_buffer.test";
  ::FILE *file = __llvm_libc::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  __llvm_libc::setbuf(file, nullptr);
  constexpr char CONTENT[] = "abcdef";
  constexpr size_t CONTENT_SIZE = sizeof(CONTENT);
  ASSERT_EQ(CONTENT_SIZE, __llvm_libc::fwrite(CONTENT, 1, CONTENT_SIZE, file));
  ASSERT_EQ(0, __llvm_libc::fclose(file));

  file = __llvm_libc::fopen(FILENAME, "r");
  __llvm_libc::setbuf(file, nullptr);
  ASSERT_FALSE(file == nullptr);
  char data[CONTENT_SIZE];
  ASSERT_EQ(__llvm_libc::fread(&data, 1, CONTENT_SIZE, file), CONTENT_SIZE);
  ASSERT_STREQ(CONTENT, data);

  // Ensure that ungetc also works.
  char unget_char = 'z';
  ASSERT_EQ(int(unget_char), __llvm_libc::ungetc(unget_char, file));
  char c;
  ASSERT_EQ(__llvm_libc::fread(&c, 1, 1, file), size_t(1));
  ASSERT_EQ(c, unget_char);

  ASSERT_EQ(0, __llvm_libc::fclose(file));
}
