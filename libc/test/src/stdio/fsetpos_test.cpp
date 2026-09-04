//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for fsetpos
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
#include "src/stdio/fsetpos.h"
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

class LlvmLibcFsetposTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  void test_with_bufmode(int bufmode) {
    constexpr char FILENAME[] = "testdata/fsetpos_bufmode.test";
    auto FILEPATH = libc_make_test_file_path(FILENAME);

    constexpr size_t BUFFER_SIZE = 1024;
    char buffer[BUFFER_SIZE];

    ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
    ASSERT_FALSE(file == nullptr);
    scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

    ASSERT_EQ(0, LIBC_NAMESPACE::setvbuf(file, buffer, bufmode, BUFFER_SIZE));

    fpos_t pos_start;
    ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_start), Succeeds(0));

    constexpr char CONTENT[] = "1234567890";
    constexpr size_t WRITE_SIZE = sizeof(CONTENT) - 1;
    ASSERT_THAT(LIBC_NAMESPACE::fwrite(CONTENT, 1, WRITE_SIZE, file),
                Succeeds(WRITE_SIZE));

    fpos_t pos_end;
    ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_end), Succeeds(0));

    // Reposition back to start
    ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));

    char data[WRITE_SIZE + 1];
    ASSERT_THAT(LIBC_NAMESPACE::fread(data, 1, WRITE_SIZE, file),
                Succeeds(WRITE_SIZE));
    data[WRITE_SIZE] = '\0';
    ASSERT_STREQ(data, CONTENT);

    // Reposition back to end
    ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_end), Succeeds(0));
    char extra;
    ASSERT_EQ(LIBC_NAMESPACE::fread(&extra, 1, 1, file), size_t(0));
  }
};

TEST_F(LlvmLibcFsetposTest, BasicRepositioningAndRoundTrip) {
  constexpr char FILENAME[] = "testdata/fsetpos_basic.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "wb+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  fpos_t pos_start;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_start), Succeeds(0));

  constexpr char DATA[] = "abcdefghij"; // 10 bytes
  constexpr size_t DATA_SIZE = sizeof(DATA) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(DATA, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));

  fpos_t pos_end;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_end), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 5, SEEK_SET), Succeeds(0));
  fpos_t pos_mid;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_mid), Succeeds(0));

  // Set position to middle (offset 5) and read 5 bytes ("fghij")
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_mid), Succeeds(0));
  char buf[6];
  ASSERT_THAT(LIBC_NAMESPACE::fread(buf, 1, 5, file), Succeeds(size_t(5)));
  buf[5] = '\0';
  ASSERT_STREQ(buf, "fghij");

  // Set position to start (offset 0) and read 5 bytes ("abcde")
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::fread(buf, 1, 5, file), Succeeds(size_t(5)));
  buf[5] = '\0';
  ASSERT_STREQ(buf, "abcde");

  // Set position to end (offset 10) and read 0 bytes
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_end), Succeeds(0));
  char extra;
  ASSERT_EQ(LIBC_NAMESPACE::fread(&extra, 1, 1, file), size_t(0));
}

TEST_F(LlvmLibcFsetposTest, ClearEOFIndicator) {
  constexpr char FILENAME[] = "testdata/fsetpos_eof.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "wb+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  fpos_t pos_start;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_start), Succeeds(0));

  constexpr char DATA[] = "hello";
  constexpr size_t DATA_SIZE = sizeof(DATA) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(DATA, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), Succeeds(0));

  char buf[DATA_SIZE];
  ASSERT_THAT(LIBC_NAMESPACE::fread(buf, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));

  // Read past end to set EOF indicator
  char extra;
  ASSERT_EQ(LIBC_NAMESPACE::fread(&extra, 1, 1, file), size_t(0));
  ASSERT_NE(LIBC_NAMESPACE::feof(file), 0);

  // ISO C §7.21.9.3: A successful call to fsetpos clears the EOF indicator
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::feof(file), 0);

  // Subsequent read succeeds from beginning
  ASSERT_THAT(LIBC_NAMESPACE::fread(buf, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));
  ASSERT_EQ(LIBC_NAMESPACE::feof(file), 0);
}

TEST_F(LlvmLibcFsetposTest, UndoUngetc) {
  constexpr char FILENAME[] = "testdata/fsetpos_ungetc.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "wb+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  constexpr char DATA[] = "0123456789";
  constexpr size_t DATA_SIZE = sizeof(DATA) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(DATA, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));

  // Seek to offset 2 and save position
  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 2, SEEK_SET), Succeeds(0));
  fpos_t pos_two;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_two), Succeeds(0));

  // Read byte at offset 2 ('2')
  char c;
  ASSERT_THAT(LIBC_NAMESPACE::fread(&c, 1, 1, file), Succeeds(size_t(1)));
  ASSERT_EQ(c, '2');

  // Push back character 'Z'
  ASSERT_EQ(LIBC_NAMESPACE::ungetc('Z', file), int('Z'));

  // ISO C §7.21.9.3: Successful fsetpos undoes any effects of ungetc
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_two), Succeeds(0));

  // Next read must return original byte '2', not pushed-back 'Z'
  ASSERT_THAT(LIBC_NAMESPACE::fread(&c, 1, 1, file), Succeeds(size_t(1)));
  ASSERT_EQ(c, '2');
}

TEST_F(LlvmLibcFsetposTest, UndoUngetwc) {
  constexpr char FILENAME[] = "testdata/fsetpos_ungetwc.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  ASSERT_GT(LIBC_NAMESPACE::fwide(file, 1), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fputwc(L'A', file), static_cast<wint_t>(L'A'));
  ASSERT_EQ(LIBC_NAMESPACE::fputwc(L'B', file), static_cast<wint_t>(L'B'));
  ASSERT_EQ(LIBC_NAMESPACE::fputwc(L'C', file), static_cast<wint_t>(L'C'));

  // Rewind to start (offset 0 with SEEK_SET is standard conforming)
  ASSERT_THAT(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), Succeeds(0));

  // Read wide character L'A' to advance stream to character L'B'
  ASSERT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'A'));

  // Save position before L'B' using standard fgetpos
  fpos_t pos_one;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_one), Succeeds(0));

  // Read wide character L'B'
  ASSERT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'B'));

  // Push back wide character L'Z'
  ASSERT_EQ(LIBC_NAMESPACE::ungetwc(L'Z', file), static_cast<wint_t>(L'Z'));

  // fsetpos undoes effects of ungetwc (test const reference as well)
  const fpos_t &pos_one_ref = pos_one;
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_one_ref), Succeeds(0));

  // Next wide read must return L'B', not L'Z'
  ASSERT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'B'));
}

TEST_F(LlvmLibcFsetposTest, WideStreamParseState) {
  constexpr char FILENAME[] = "testdata/fsetpos_wide_state.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  ASSERT_GT(LIBC_NAMESPACE::fwide(file, 1), 0);

  fpos_t pos_start;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_start), Succeeds(0));
  ASSERT_NE(LIBC_NAMESPACE::mbsinit(&pos_start.__state), 0);

  // Write multi-byte wide character: L'¢' (0xC2, 0xA2 in UTF-8)
  ASSERT_EQ(LIBC_NAMESPACE::fputwc(L'¢', file), static_cast<wint_t>(L'¢'));

  // Reposition to beginning with fsetpos
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));

  // Stream orientation is preserved (remains wide-oriented)
  ASSERT_GT(LIBC_NAMESPACE::fwide(file, 0), 0);

  // Position and parse state are restored
  fpos_t current_pos;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &current_pos), Succeeds(0));
  ASSERT_NE(LIBC_NAMESPACE::mbsinit(&current_pos.__state), 0);
  ASSERT_EQ(current_pos.__pos, off_t(0));

  // Read back wide character
  ASSERT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'¢'));
}

TEST_F(LlvmLibcFsetposTest, UpdateStreamDirectionSwitching) {
  constexpr char FILENAME[] = "testdata/fsetpos_direction.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  fpos_t pos_start;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_start), Succeeds(0));

  // Write -> fsetpos -> Read without fflush
  constexpr char WRITE_DATA[] = "12345";
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(WRITE_DATA, 1, 5, file),
              Succeeds(size_t(5)));

  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));

  char read_buf[6];
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_buf, 1, 5, file), Succeeds(size_t(5)));
  read_buf[5] = '\0';
  ASSERT_STREQ(read_buf, WRITE_DATA);

  // Read -> fsetpos -> Write
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));
  constexpr char OVERWRITE[] = "abcde";
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(OVERWRITE, 1, 5, file),
              Succeeds(size_t(5)));

  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_buf, 1, 5, file), Succeeds(size_t(5)));
  read_buf[5] = '\0';
  ASSERT_STREQ(read_buf, OVERWRITE);
}

TEST_F(LlvmLibcFsetposTest, FlushDirtyBufferOnSeek) {
  constexpr char FILENAME[] = "testdata/fsetpos_flush.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  constexpr size_t BUFFER_SIZE = 1024;
  char buffer[BUFFER_SIZE];

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  ASSERT_EQ(0, LIBC_NAMESPACE::setvbuf(file, buffer, _IOFBF, BUFFER_SIZE));

  fpos_t pos_start;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_start), Succeeds(0));

  constexpr char DATA[] = "unflushed_data";
  constexpr size_t DATA_SIZE = sizeof(DATA) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(DATA, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));

  // fsetpos flushes unwritten buffer data before repositioning
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));

  char read_buf[DATA_SIZE + 1];
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_buf, 1, DATA_SIZE, file),
              Succeeds(DATA_SIZE));
  read_buf[DATA_SIZE] = '\0';
  ASSERT_STREQ(read_buf, DATA);
}

TEST_F(LlvmLibcFsetposTest, ErrorIndicatorPreserved) {
  constexpr char FILENAME[] = "testdata/fsetpos_ferror.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  // Open file in write-only mode
  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  fpos_t pos_start;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_start), Succeeds(0));

  // Attempting to read from write-only stream sets ferror
  char c;
  ASSERT_EQ(LIBC_NAMESPACE::fread(&c, 1, 1, file), size_t(0));
  ASSERT_NE(LIBC_NAMESPACE::ferror(file), 0);
  libc_errno = 0; // Clear errno from expected read failure

  // ISO C §7.21.9.3: fsetpos does not clear the error indicator
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));
  ASSERT_NE(LIBC_NAMESPACE::ferror(file), 0);
}

TEST_F(LlvmLibcFsetposTest, ErrnoPreservedOnSuccess) {
  constexpr char FILENAME[] = "testdata/fsetpos_errno.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "wb+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  fpos_t pos;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos), Succeeds(0));

  // Pre-set errno to non-zero value
  libc_errno = 42;

  ASSERT_EQ(LIBC_NAMESPACE::fsetpos(file, &pos), 0);
  // ISO C / POSIX: errno must remain unaltered on success
  ASSERT_EQ(static_cast<int>(libc_errno), 42);

  libc_errno = 0;
}

TEST_F(LlvmLibcFsetposTest, AppendMode) {
  constexpr char FILENAME[] = "testdata/fsetpos_append.test";
  auto FILEPATH = libc_make_test_file_path(FILENAME);

  ::FILE *init_file = LIBC_NAMESPACE::fopen(FILEPATH, "w");
  ASSERT_FALSE(init_file == nullptr);
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(init_file));

  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "ab+");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

  fpos_t pos_start;
  ASSERT_THAT(LIBC_NAMESPACE::fgetpos(file, &pos_start), Succeeds(0));

  constexpr char DATA1[] = "hello";
  constexpr size_t SIZE1 = sizeof(DATA1) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(DATA1, 1, SIZE1, file), Succeeds(SIZE1));

  // In append mode, reading can occur at any repositioned location
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));
  char buf[SIZE1 + 1];
  ASSERT_THAT(LIBC_NAMESPACE::fread(buf, 1, SIZE1, file), Succeeds(SIZE1));
  buf[SIZE1] = '\0';
  ASSERT_STREQ(buf, DATA1);

  // Reposition to start before writing
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));

  // ISO C §7.21.5.3: In append mode, all writes are forced to EOF
  constexpr char DATA2[] = "world";
  constexpr size_t SIZE2 = sizeof(DATA2) - 1;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(DATA2, 1, SIZE2, file), Succeeds(SIZE2));

  // Read back all content from start
  ASSERT_THAT(LIBC_NAMESPACE::fsetpos(file, &pos_start), Succeeds(0));
  constexpr size_t TOTAL_SIZE = SIZE1 + SIZE2;
  char total_buf[TOTAL_SIZE + 1];
  ASSERT_THAT(LIBC_NAMESPACE::fread(total_buf, 1, TOTAL_SIZE, file),
              Succeeds(TOTAL_SIZE));
  total_buf[TOTAL_SIZE] = '\0';
  ASSERT_STREQ(total_buf, "helloworld");
}

TEST_F(LlvmLibcFsetposTest, BufferingModes) {
  test_with_bufmode(_IOFBF);
  test_with_bufmode(_IOLBF);
  test_with_bufmode(_IONBF);
}
