//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getline.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/FILE.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/libc_errno.h"
#include "src/stdio/clearerr.h"
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fmemopen.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fseek.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/getline.h"
#include "src/stdio/remove.h"
#include "src/stdio/setvbuf.h"
#include "src/stdio/ungetc.h"
#include "src/wchar/fwide.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::scope_exit;
using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

class LlvmLibcGetlineTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  void test_long_lines_with_bufmode(int bufmode) {
    constexpr char FILENAME[] = "getline_bufmode.test";
    auto FILEPATH = libc_make_test_file_path(FILENAME);

    ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
    ASSERT_NE(file, nullptr);
    scope_exit close_file([&] {
      ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0));
      ASSERT_THAT(LIBC_NAMESPACE::remove(FILEPATH), Succeeds(0));
    });

    constexpr char CHARS[] = "abcXYZ0123_!@#";
    constexpr size_t LINE1_LEN = 2500;
    char expected1[LINE1_LEN + 1];
    for (size_t i = 0; i < LINE1_LEN - 1; ++i)
      expected1[i] = CHARS[i % (sizeof(CHARS) - 1)];
    expected1[LINE1_LEN - 1] = '\n';
    expected1[LINE1_LEN] = '\0';

    constexpr char EXPECTED2[] = "trailing line without newline";
    constexpr size_t LINE2_LEN = sizeof(EXPECTED2) - 1;

    ASSERT_THAT(LIBC_NAMESPACE::fwrite(expected1, 1, LINE1_LEN, file),
                Succeeds(LINE1_LEN));
    ASSERT_THAT(LIBC_NAMESPACE::fwrite(EXPECTED2, 1, LINE2_LEN, file),
                Succeeds(LINE2_LEN));
    ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), Succeeds(0));

    char stream_buf[32];
    if (bufmode == _IONBF) {
      ASSERT_EQ(LIBC_NAMESPACE::setvbuf(file, nullptr, _IONBF, 0), 0);
    } else {
      ASSERT_EQ(LIBC_NAMESPACE::setvbuf(file, stream_buf, bufmode,
                                        sizeof(stream_buf)),
                0);
    }

    char *line = nullptr;
    size_t cap = 0;
    scope_exit free_line([&] { ::free(line); });

    ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, file),
                Succeeds(ssize_t(LINE1_LEN)));
    ASSERT_NE(line, nullptr);
    EXPECT_GE(cap, LINE1_LEN + 1);
    EXPECT_STREQ(line, expected1);
    EXPECT_EQ(LIBC_NAMESPACE::feof(file), 0);
    EXPECT_EQ(LIBC_NAMESPACE::ferror(file), 0);

    ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, file),
                Succeeds(ssize_t(LINE2_LEN)));
    EXPECT_STREQ(line, EXPECTED2);
    EXPECT_NE(LIBC_NAMESPACE::feof(file), 0);
    EXPECT_EQ(LIBC_NAMESPACE::ferror(file), 0);

    ASSERT_EQ(LIBC_NAMESPACE::getline(&line, &cap, file), ssize_t(-1));
    EXPECT_NE(LIBC_NAMESPACE::feof(file), 0);
    EXPECT_EQ(LIBC_NAMESPACE::ferror(file), 0);
  }
};

TEST_F(LlvmLibcGetlineTest, AllocatesWhenLineptrIsNull) {
  char content[] = "hello world\n";
  ::FILE *stream = LIBC_NAMESPACE::fmemopen(content, sizeof(content) - 1, "r");
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(stream), Succeeds(0)); });

  char *line = nullptr;
  size_t cap = 0;
  scope_exit free_line([&] { ::free(line); });

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, stream),
              Succeeds(ssize_t(12)));
  ASSERT_NE(line, nullptr);
  EXPECT_GE(cap, size_t(13));
  EXPECT_STREQ(line, "hello world\n");
  EXPECT_EQ(LIBC_NAMESPACE::feof(stream), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(stream), 0);

  ASSERT_EQ(LIBC_NAMESPACE::getline(&line, &cap, stream), ssize_t(-1));
  EXPECT_NE(LIBC_NAMESPACE::feof(stream), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(stream), 0);
}

TEST_F(LlvmLibcGetlineTest, NullLineptrIgnoresNonZeroCapacity) {
  char content[] = "short line\n";
  ::FILE *stream = LIBC_NAMESPACE::fmemopen(content, sizeof(content) - 1, "r");
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(stream), Succeeds(0)); });

  char *line = nullptr;
  size_t cap = 1024;
  scope_exit free_line([&] { ::free(line); });

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, stream),
              Succeeds(ssize_t(11)));
  ASSERT_NE(line, nullptr);
  EXPECT_GE(cap, size_t(12));
  EXPECT_STREQ(line, "short line\n");
}

TEST_F(LlvmLibcGetlineTest, ReusesSufficientCallerBuffer) {
  char content[] = "fits in buffer\n";
  ::FILE *stream = LIBC_NAMESPACE::fmemopen(content, sizeof(content) - 1, "r");
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(stream), Succeeds(0)); });

  size_t cap = 64;
  char *line = static_cast<char *>(::malloc(cap));
  ASSERT_NE(line, nullptr);
  scope_exit free_line([&] { ::free(line); });
  char *original_ptr = line;

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, stream),
              Succeeds(ssize_t(15)));
  EXPECT_EQ(line, original_ptr);
  EXPECT_EQ(cap, size_t(64));
  EXPECT_STREQ(line, "fits in buffer\n");
}

TEST_F(LlvmLibcGetlineTest, GrowsSmallAndZeroSizedCallerBuffers) {
  constexpr char CHARS[] = "pqrs789_+-=";
  constexpr size_t LEN = 300;
  char content[LEN];
  for (size_t i = 0; i < LEN - 1; ++i)
    content[i] = CHARS[i % (sizeof(CHARS) - 1)];
  content[LEN - 1] = '\n';

  // Case 1: 1-byte malloc'd buffer (needs space for char + NUL immediately).
  {
    ::FILE *stream = LIBC_NAMESPACE::fmemopen(content, LEN, "r");
    ASSERT_NE(stream, nullptr);
    scope_exit close_stream(
        [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(stream), Succeeds(0)); });

    size_t cap = 1;
    char *line = static_cast<char *>(::malloc(cap));
    ASSERT_NE(line, nullptr);
    scope_exit free_line([&] { ::free(line); });

    ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, stream),
                Succeeds(ssize_t(LEN)));
    ASSERT_NE(line, nullptr);
    EXPECT_GE(cap, LEN + 1);
    EXPECT_EQ(string_view(line, LEN), string_view(content, LEN));
    EXPECT_EQ(line[LEN], '\0');
  }

  // Case 2: non-null malloc(0) pointer with *n == 0.
  {
    ::FILE *stream = LIBC_NAMESPACE::fmemopen(content, LEN, "r");
    ASSERT_NE(stream, nullptr);
    scope_exit close_stream(
        [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(stream), Succeeds(0)); });

    size_t cap = 0;
    char *line = static_cast<char *>(::malloc(0));
    scope_exit free_line([&] { ::free(line); });

    ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, stream),
                Succeeds(ssize_t(LEN)));
    ASSERT_NE(line, nullptr);
    EXPECT_GE(cap, LEN + 1);
    EXPECT_EQ(string_view(line, LEN), string_view(content, LEN));
    EXPECT_EQ(line[LEN], '\0');
  }
}

TEST_F(LlvmLibcGetlineTest, MultipleLinesEmptyLineAndNoTrailingNewline) {
  char content[] = "first\n\nthird without newline";
  ::FILE *stream = LIBC_NAMESPACE::fmemopen(content, sizeof(content) - 1, "r");
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(stream), Succeeds(0)); });

  char *line = nullptr;
  size_t cap = 0;
  scope_exit free_line([&] { ::free(line); });

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, stream),
              Succeeds(ssize_t(6)));
  EXPECT_STREQ(line, "first\n");
  EXPECT_EQ(LIBC_NAMESPACE::feof(stream), 0);

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, stream),
              Succeeds(ssize_t(1)));
  EXPECT_STREQ(line, "\n");
  EXPECT_EQ(LIBC_NAMESPACE::feof(stream), 0);

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, stream),
              Succeeds(ssize_t(21)));
  EXPECT_STREQ(line, "third without newline");
  EXPECT_NE(LIBC_NAMESPACE::feof(stream), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(stream), 0);

  ASSERT_EQ(LIBC_NAMESPACE::getline(&line, &cap, stream), ssize_t(-1));
  EXPECT_NE(LIBC_NAMESPACE::feof(stream), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(stream), 0);
}

TEST_F(LlvmLibcGetlineTest, EmptyFilePreservesErrnoOnEof) {
  constexpr char FILENAME[] = "getline_empty.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_NE(file, nullptr);
  scope_exit close_file([&] {
    ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::remove(FILEPATH), Succeeds(0));
  });

  char *line = nullptr;
  size_t cap = 0;
  scope_exit free_line([&] { ::free(line); });

  libc_errno = 42;
  ASSERT_EQ(LIBC_NAMESPACE::getline(&line, &cap, file), ssize_t(-1));
  ASSERT_ERRNO_EQ(42);
  EXPECT_NE(LIBC_NAMESPACE::feof(file), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(file), 0);
}

TEST_F(LlvmLibcGetlineTest, EmbeddedNullBytesAndHighBitBytes) {
  constexpr char FILENAME[] = "getline_binary.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_NE(file, nullptr);
  scope_exit close_file([&] {
    ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::remove(FILEPATH), Succeeds(0));
  });

  constexpr char DATA[] = "ab\0cd\xff"
                          "ef\n\0tail";
  constexpr size_t DATA_LEN = sizeof(DATA) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(DATA, 1, DATA_LEN, file),
              Succeeds(DATA_LEN));
  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), Succeeds(0));

  char *line = nullptr;
  size_t cap = 0;
  scope_exit free_line([&] { ::free(line); });

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, file), Succeeds(ssize_t(9)));
  EXPECT_EQ(string_view(line, 9), string_view("ab\0cd\xff"
                                              "ef\n",
                                              9));
  EXPECT_EQ(line[9], '\0');

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, file), Succeeds(ssize_t(5)));
  EXPECT_EQ(string_view(line, 5), string_view("\0tail", 5));
  EXPECT_EQ(line[5], '\0');
}

TEST_F(LlvmLibcGetlineTest, BufferingModes) {
  test_long_lines_with_bufmode(_IOFBF);
  test_long_lines_with_bufmode(_IOLBF);
  test_long_lines_with_bufmode(_IONBF);
}

TEST_F(LlvmLibcGetlineTest, StickyEofClearedByFseekAndUngetc) {
  constexpr char FILENAME[] = "getline_sticky_eof.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *reader = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_NE(reader, nullptr);
  ::FILE *appender = LIBC_NAMESPACE::fopen(FILEPATH, "a");
  ASSERT_NE(appender, nullptr);
  scope_exit close_files([&] {
    if (appender != nullptr)
      ASSERT_THAT(LIBC_NAMESPACE::fclose(appender), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::fclose(reader), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::remove(FILEPATH), Succeeds(0));
  });

  char *line = nullptr;
  size_t cap = 0;
  scope_exit free_line([&] { ::free(line); });

  // Hit EOF on empty file.
  ASSERT_EQ(LIBC_NAMESPACE::getline(&line, &cap, reader), ssize_t(-1));
  ASSERT_NE(LIBC_NAMESPACE::feof(reader), 0);

  // Append new data externally; sticky EOF on reader must still return -1.
  constexpr char MORE[] = "appended\n";
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(MORE, 1, sizeof(MORE) - 1, appender),
              Succeeds(sizeof(MORE) - 1));
  ASSERT_THAT(LIBC_NAMESPACE::fclose(appender), Succeeds(0));
  appender = nullptr;

  ASSERT_EQ(LIBC_NAMESPACE::getline(&line, &cap, reader), ssize_t(-1));
  EXPECT_NE(LIBC_NAMESPACE::feof(reader), 0);

  // Positioning operation clears sticky EOF and allows reading appended line.
  ASSERT_THAT(LIBC_NAMESPACE::fseek(reader, 0, SEEK_CUR), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, reader),
              Succeeds(ssize_t(9)));
  EXPECT_STREQ(line, "appended\n");

  // Hit EOF again, then push back a character with ungetc.
  ASSERT_EQ(LIBC_NAMESPACE::getline(&line, &cap, reader), ssize_t(-1));
  ASSERT_NE(LIBC_NAMESPACE::feof(reader), 0);
  ASSERT_EQ(LIBC_NAMESPACE::ungetc('Z', reader), int('Z'));
  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, reader),
              Succeeds(ssize_t(1)));
  EXPECT_STREQ(line, "Z");
}

TEST_F(LlvmLibcGetlineTest, InvalidArgumentsSetErrnoAndStreamError) {
  char content[] = "abc\n";
  ::FILE *stream = LIBC_NAMESPACE::fmemopen(content, sizeof(content) - 1, "r");
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(stream), Succeeds(0)); });

  size_t cap = 0;
  ASSERT_THAT(LIBC_NAMESPACE::getline(nullptr, &cap, stream),
              Fails(EINVAL, ssize_t(-1)));
  EXPECT_NE(LIBC_NAMESPACE::ferror(stream), 0);

  LIBC_NAMESPACE::clearerr(stream);
  char *line = nullptr;
  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, nullptr, stream),
              Fails(EINVAL, ssize_t(-1)));
  EXPECT_NE(LIBC_NAMESPACE::ferror(stream), 0);
  EXPECT_EQ(line, nullptr);
}

TEST_F(LlvmLibcGetlineTest, WriteOnlyStreamFailsWithEbadf) {
  constexpr char FILENAME[] = "getline_write_only.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w");
  ASSERT_NE(file, nullptr);
  scope_exit close_file([&] {
    ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::remove(FILEPATH), Succeeds(0));
  });

  char *line = nullptr;
  size_t cap = 0;
  scope_exit free_line([&] { ::free(line); });

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, file),
              Fails(EBADF, ssize_t(-1)));
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);
  EXPECT_EQ(line, nullptr);
}

TEST_F(LlvmLibcGetlineTest, WideOrientedStreamFailsWithEinval) {
  constexpr char FILENAME[] = "getline_wide.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_NE(file, nullptr);
  scope_exit close_file([&] {
    ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::remove(FILEPATH), Succeeds(0));
  });

  ASSERT_GT(LIBC_NAMESPACE::fwide(file, 1), 0);

  char *line = nullptr;
  size_t cap = 0;
  scope_exit free_line([&] { ::free(line); });

  ASSERT_THAT(LIBC_NAMESPACE::getline(&line, &cap, file),
              Fails(EINVAL, ssize_t(-1)));
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);
  EXPECT_EQ(line, nullptr);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcGetlineTest, NullStreamCrash) {
  EXPECT_DEATH(
      [] {
        char *line = nullptr;
        size_t cap = 0;
        LIBC_NAMESPACE::getline(&line, &cap, nullptr);
      },
      WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
