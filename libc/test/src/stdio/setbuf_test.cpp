//===-- Unittests for setbuf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdio_macros.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/setbuf.h"
#include "src/stdio/ungetc.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSetbufTest, DefaultBufsize) {
  // The idea in this test is to change the buffer after opening a file and
  // ensure that read and write work as expected.
  constexpr char FILENAME[] =
      APPEND_LIBC_TEST("testdata/setbuf_test_default_bufsize.test");
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  char buffer[BUFSIZ];
  LIBC_NAMESPACE::setbuf(file, buffer);
  constexpr char CONTENT[] = "abcdef";
  constexpr size_t CONTENT_SIZE = sizeof(CONTENT);
  ASSERT_EQ(CONTENT_SIZE,
            LIBC_NAMESPACE::fwrite(CONTENT, 1, CONTENT_SIZE, file));
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  LIBC_NAMESPACE::setbuf(file, buffer);
  ASSERT_FALSE(file == nullptr);
  char data[CONTENT_SIZE];
  ASSERT_EQ(LIBC_NAMESPACE::fread(&data, 1, CONTENT_SIZE, file), CONTENT_SIZE);
  ASSERT_STREQ(CONTENT, data);
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));
}

TEST(LlvmLibcSetbufTest, NullBuffer) {
  // The idea in this test is that we set a null buffer and ensure that
  // everything works correctly.
  constexpr char FILENAME[] =
      APPEND_LIBC_TEST("testdata/setbuf_test_null_buffer.test");
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  LIBC_NAMESPACE::setbuf(file, nullptr);
  constexpr char CONTENT[] = "abcdef";
  constexpr size_t CONTENT_SIZE = sizeof(CONTENT);
  ASSERT_EQ(CONTENT_SIZE,
            LIBC_NAMESPACE::fwrite(CONTENT, 1, CONTENT_SIZE, file));
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  LIBC_NAMESPACE::setbuf(file, nullptr);
  ASSERT_FALSE(file == nullptr);
  char data[CONTENT_SIZE];
  ASSERT_EQ(LIBC_NAMESPACE::fread(&data, 1, CONTENT_SIZE, file), CONTENT_SIZE);
  ASSERT_STREQ(CONTENT, data);

  // Ensure that ungetc also works.
  char unget_char = 'z';
  ASSERT_EQ(int(unget_char), LIBC_NAMESPACE::ungetc(unget_char, file));
  char c;
  ASSERT_EQ(LIBC_NAMESPACE::fread(&c, 1, 1, file), size_t(1));
  ASSERT_EQ(c, unget_char);

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));
}
