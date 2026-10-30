//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setvbuf.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/clearerr.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/setvbuf.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

#include "hdr/stdio_macros.h"

using LlvmLibcSetvbufTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetvbufTest, SetNBFBuffer) {
  // The idea in this test is that we open a file for writing and reading, and
  // then set a NBF buffer to the write handle. Since it is NBF, the data
  // written using the write handle should be immediately readable by the read
  // handle.
  constexpr char FILENAME[] = "testdata/setvbuf_nbf.test";

  ::FILE *fw = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(fw == nullptr);
  char buffer[BUFSIZ];
  ASSERT_EQ(LIBC_NAMESPACE::setvbuf(fw, buffer, _IONBF, BUFSIZ), 0);

  ::FILE *fr = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(fr == nullptr);

  constexpr char CONTENT[] = "abcdef";
  constexpr size_t CONTENT_SIZE = sizeof(CONTENT);
  for (size_t i = 0; i < CONTENT_SIZE; ++i) {
    ASSERT_EQ(size_t(1), LIBC_NAMESPACE::fwrite(CONTENT + i, 1, 1, fw));
    char c;
    ASSERT_EQ(size_t(1), LIBC_NAMESPACE::fread(&c, 1, 1, fr));
    ASSERT_EQ(c, CONTENT[i]);
  }

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(fw));
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(fr));

  // Make sure NBF buffer has no effect for reading.
  fr = LIBC_NAMESPACE::fopen(FILENAME, "r");
  char data[CONTENT_SIZE];
  ASSERT_EQ(LIBC_NAMESPACE::setvbuf(fr, buffer, _IONBF, BUFSIZ), 0);
  ASSERT_EQ(CONTENT_SIZE, LIBC_NAMESPACE::fread(data, 1, CONTENT_SIZE, fr));
  ASSERT_STREQ(CONTENT, data);
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(fr));
}

TEST_F(LlvmLibcSetvbufTest, SetLBFBuffer) {
  // The idea in this test is that we open a file for writing and reading, and
  // then set a LBF buffer to the write handle. Since it is LBF, the data
  // written using the write handle should be available right after a '\n' is
  // written.
  constexpr char FILENAME[] = "testdata/setvbuf_lbf.test";

  ::FILE *fw = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(fw == nullptr);
  char buffer[BUFSIZ];
  ASSERT_EQ(LIBC_NAMESPACE::setvbuf(fw, buffer, _IOLBF, BUFSIZ), 0);

  ::FILE *fr = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(fr == nullptr);

  constexpr char CONTENT[] = "abcdef\n";
  constexpr size_t CONTENT_SIZE = sizeof(CONTENT);
  ASSERT_EQ(CONTENT_SIZE, LIBC_NAMESPACE::fwrite(CONTENT, 1, CONTENT_SIZE, fw));

  // Note that CONTENT_SIZE worth of data written also includes the
  // null-terminator '\0'. But, since it is after the new line character,
  // it should not be availabe for reading.
  char data[CONTENT_SIZE];
  ASSERT_EQ(CONTENT_SIZE - 1, LIBC_NAMESPACE::fread(data, 1, CONTENT_SIZE, fr));
  char c;
  ASSERT_EQ(size_t(0), LIBC_NAMESPACE::fread(&c, 1, 1, fr));

  data[CONTENT_SIZE - 1] = '\0';
  ASSERT_STREQ(CONTENT, data);

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(fw));
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(fr));

  // Make sure LBF buffer has no effect for reading.
  fr = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_EQ(LIBC_NAMESPACE::setvbuf(fr, buffer, _IOLBF, BUFSIZ), 0);
  ASSERT_EQ(CONTENT_SIZE, LIBC_NAMESPACE::fread(data, 1, CONTENT_SIZE, fr));
  ASSERT_STREQ(CONTENT, data);
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(fr));
}

TEST_F(LlvmLibcSetvbufTest, NullBufferAsksForOne) {
  // A null buffer asks the library to provide one, so the stream stays
  // buffered in the mode asked for whatever the size. Data written to a fully
  // buffered stream is not readable through another handle until it is
  // flushed.
  constexpr char FILENAME[] = "testdata/setvbuf_null_buffer.test";
  constexpr char CONTENT[] = "abcdef";
  constexpr size_t CONTENT_SIZE = sizeof(CONTENT) - 1;
  const size_t sizes[] = {0, 16};

  for (size_t size : sizes) {
    ::FILE *fw = LIBC_NAMESPACE::fopen(FILENAME, "w");
    ASSERT_FALSE(fw == nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::setvbuf(fw, nullptr, _IOFBF, size), 0);

    ::FILE *fr = LIBC_NAMESPACE::fopen(FILENAME, "r");
    ASSERT_FALSE(fr == nullptr);

    ASSERT_EQ(CONTENT_SIZE,
              LIBC_NAMESPACE::fwrite(CONTENT, 1, CONTENT_SIZE, fw));
    char data[CONTENT_SIZE + 1] = {};
    ASSERT_EQ(size_t(0), LIBC_NAMESPACE::fread(data, 1, CONTENT_SIZE, fr));

    ASSERT_EQ(0, LIBC_NAMESPACE::fflush(fw));
    LIBC_NAMESPACE::clearerr(fr);
    ASSERT_EQ(CONTENT_SIZE, LIBC_NAMESPACE::fread(data, 1, CONTENT_SIZE, fr));
    ASSERT_STREQ(CONTENT, data);

    ASSERT_EQ(0, LIBC_NAMESPACE::fclose(fw));
    ASSERT_EQ(0, LIBC_NAMESPACE::fclose(fr));
  }
}

TEST(LlvmLibcSetbufTest, InvalidBufferMode) {
  constexpr char FILENAME[] = "testdata/setvbuf_invalid_bufmode.test";
  ::FILE *f = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(f == nullptr);
  char buf[BUFSIZ];
  ASSERT_NE(LIBC_NAMESPACE::setvbuf(f, buf, _IOFBF + _IOLBF + _IONBF, BUFSIZ),
            0);
  ASSERT_ERRNO_EQ(EINVAL);

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(f));
}
