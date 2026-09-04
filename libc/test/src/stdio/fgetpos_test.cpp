//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for fgetpos
///
//===----------------------------------------------------------------------===//

#include "hdr/stdio_macros.h"
#include "hdr/types/fpos_t.h"
#include "hdr/types/wint_t.h"
#include "src/__support/CPP/scope.h"
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fgetpos.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fseek.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/setvbuf.h"
#include "src/stdio/ungetc.h"
#include "src/wchar/fgetwc.h"
#include "src/wchar/fputwc.h"
#include "src/wchar/fwide.h"
#include "src/wchar/mbsinit.h"
#include "src/wchar/ungetwc.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::scope_exit;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

class LlvmLibcFgetposTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  void test_with_bufmode(int bufmode) {
    constexpr char FILENAME[] = "testdata/fgetpos_bufmode.test";
    auto FILEPATH = libc_make_test_file_path(FILENAME);

    constexpr size_t BUFFER_SIZE = 1024;
    char buffer[BUFFER_SIZE];

    ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
    ASSERT_FALSE(file == nullptr);
    scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

    ASSERT_EQ(0, LIBC_NAMESPACE::setvbuf(file, buffer, bufmode, BUFFER_SIZE));

    constexpr char CONTENT[] = "12\n345\n6789";
    constexpr size_t WRITE_SIZE = sizeof(CONTENT) - 1;
    ASSERT_THAT(LIBC_NAMESPACE::fwrite(CONTENT, 1, WRITE_SIZE, file),
                Succeeds(WRITE_SIZE));

    fpos_t pos;
    ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
    ASSERT_EQ(pos.__pos, off_t(WRITE_SIZE));

    ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), Succeeds(0));
    constexpr size_t READ_SIZE = WRITE_SIZE / 2;
    char data[READ_SIZE];
    ASSERT_THAT(LIBC_NAMESPACE::fread(data, 1, READ_SIZE, file),
                Succeeds(READ_SIZE));
    ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
    ASSERT_EQ(pos.__pos, off_t(READ_SIZE));
  }
};

TEST_F(LlvmLibcFgetposTest, NewlyOpenedFile) {
  constexpr char FILENAME[] = "testdata/fgetpos_newly_opened.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  fpos_t pos;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(0));
}

TEST_F(LlvmLibcFgetposTest, WriteAndRead) {
  constexpr char FILENAME[] = "testdata/fgetpos_write_read.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  constexpr char FIRST_DATA[] = "1234567890"; // 10 bytes
  constexpr size_t FIRST_SIZE = sizeof(FIRST_DATA) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(FIRST_DATA, 1, FIRST_SIZE, file),
              Succeeds(FIRST_SIZE));

  fpos_t pos;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(FIRST_SIZE));

  constexpr char SECOND_DATA[] = "abcde"; // 5 bytes
  constexpr size_t SECOND_SIZE = sizeof(SECOND_DATA) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(SECOND_DATA, 1, SECOND_SIZE, file),
              Succeeds(SECOND_SIZE));

  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(FIRST_SIZE + SECOND_SIZE));

  // Seek to offset 4
  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 4, SEEK_SET), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(4));

  // Read 3 bytes
  char read_buf[4];
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_buf, 1, 3, file), Succeeds(size_t(3)));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(7));
}

TEST_F(LlvmLibcFgetposTest, UngetcEffectBinaryStream) {
  constexpr char FILENAME[] = "testdata/fgetpos_ungetc_bin.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  // ISO C §7.21.7.10 guarantees decrement by 1 specifically for binary streams.
  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "wb+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  constexpr char DATA[] = "abcdef";
  constexpr size_t DATA_SIZE = sizeof(DATA) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(DATA, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));

  // Rewind to beginning
  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), Succeeds(0));

  char read_buf[4];
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_buf, 1, 3, file), Succeeds(size_t(3)));

  fpos_t pos;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(3));

  // Pushing back a character decrements the file position indicator by 1
  ASSERT_EQ(LIBC_NAMESPACE::ungetc(read_buf[2], file), int(read_buf[2]));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(2));

  // Reading back the character restores position
  char c;
  ASSERT_THAT(LIBC_NAMESPACE::fread(&c, 1, 1, file), Succeeds(size_t(1)));
  ASSERT_EQ(c, read_buf[2]);
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(3));
}

TEST_F(LlvmLibcFgetposTest, WideStream) {
  constexpr char FILENAME[] = "testdata/fgetpos_wide.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  // Orient stream to wide
  ASSERT_GT(LIBC_NAMESPACE::fwide(file, 1), 0);

  fpos_t pos;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(0));
  // Verify initial parse state is recorded
  ASSERT_NE(LIBC_NAMESPACE::mbsinit(&pos.__state), 0);

  // Write ASCII wide char (1 byte in UTF-8)
  ASSERT_EQ(LIBC_NAMESPACE::fputwc(L'A', file), static_cast<wint_t>(L'A'));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(1));

  // Write multi-byte wide char: L'¢' (2 bytes in UTF-8: 0xC2, 0xA2)
  ASSERT_EQ(LIBC_NAMESPACE::fputwc(L'¢', file), static_cast<wint_t>(L'¢'));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(3));

  // Seek to start
  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(0));

  // Read back first wide char
  ASSERT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'A'));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(1));

  // Read back second wide char (multi-byte)
  ASSERT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'¢'));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(3));

  // Push back wide character and then read back to verify restoration
  ASSERT_EQ(LIBC_NAMESPACE::ungetwc(L'¢', file), static_cast<wint_t>(L'¢'));
  ASSERT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'¢'));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(3));
}

TEST_F(LlvmLibcFgetposTest, AtEOF) {
  constexpr char FILENAME[] = "testdata/fgetpos_eof.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  constexpr char DATA[] = "hello";
  constexpr size_t DATA_SIZE = sizeof(DATA) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(DATA, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), Succeeds(0));

  char buf[DATA_SIZE + 1];
  ASSERT_THAT(LIBC_NAMESPACE::fread(buf, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));

  // Reading past end sets EOF indicator
  char extra;
  ASSERT_EQ(LIBC_NAMESPACE::fread(&extra, 1, 1, file), size_t(0));
  ASSERT_NE(LIBC_NAMESPACE::feof(file), 0);

  // fgetpos at EOF must succeed, report file size, and preserve EOF indicator
  fpos_t pos;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(DATA_SIZE));
  ASSERT_NE(LIBC_NAMESPACE::feof(file), 0);
}

TEST_F(LlvmLibcFgetposTest, AppendMode) {
  constexpr char FILENAME[] = "testdata/fgetpos_append.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *init_file = LIBC_NAMESPACE::fopen(FILEPATH, "w");
  ASSERT_FALSE(init_file == nullptr);
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(init_file));

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "ab+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  constexpr char FIRST[] = "12345";
  constexpr size_t FIRST_SIZE = sizeof(FIRST) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(FIRST, 1, FIRST_SIZE, file),
              Succeeds(FIRST_SIZE));

  fpos_t pos;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(FIRST_SIZE));

  // Seek to beginning
  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(0));

  // In append mode, write is always positioned at EOF
  constexpr char SECOND[] = "67890";
  constexpr size_t SECOND_SIZE = sizeof(SECOND) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(SECOND, 1, SECOND_SIZE, file),
              Succeeds(SECOND_SIZE));

  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));
  ASSERT_EQ(pos.__pos, off_t(FIRST_SIZE + SECOND_SIZE));
}

TEST_F(LlvmLibcFgetposTest, ErrnoPreservedOnSuccess) {
  constexpr char FILENAME[] = "testdata/fgetpos_errno.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  // Pre-set errno to non-zero value
  libc_errno = 42;

  fpos_t pos;
  ASSERT_EQ(LIBC_NAMESPACE::fgetpos(file, &pos), 0);
  ASSERT_EQ(pos.__pos, off_t(0));
  // ISO C / POSIX: errno must remain unaltered on success
  ASSERT_EQ(static_cast<int>(libc_errno), 42);

  // Restore errno for test fixture cleanup
  libc_errno = 0;
}

TEST_F(LlvmLibcFgetposTest, BufferingModes) {
  test_with_bufmode(_IOFBF);
  test_with_bufmode(_IOLBF);
  test_with_bufmode(_IONBF);
}
