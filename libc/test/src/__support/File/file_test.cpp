//===-- Unittests for platform independent file class ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"
#include "utils/UnitTest/MemoryMatcher.h"
#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

using ModeFlags = __llvm_libc::File::ModeFlags;
using MemoryView = __llvm_libc::memory::testing::MemoryView;

class StringFile : public __llvm_libc::File {
  static constexpr size_t SIZE = 512;
  size_t pos;
  char str[SIZE] = {0};
  size_t eof_marker;
  bool write_append;

  static size_t str_read(__llvm_libc::File *f, void *data, size_t len);
  static size_t str_write(__llvm_libc::File *f, const void *data, size_t len);
  static int str_seek(__llvm_libc::File *f, long offset, int whence);
  static int str_close(__llvm_libc::File *f) { return 0; }
  static int str_flush(__llvm_libc::File *f) { return 0; }

public:
  explicit StringFile(char *buffer, size_t buflen, int bufmode, bool owned,
                      ModeFlags modeflags)
      : __llvm_libc::File(&str_write, &str_read, &str_seek, &str_close,
                          &str_flush, buffer, buflen, bufmode, owned,
                          modeflags),
        pos(0), eof_marker(0), write_append(false) {
    if (modeflags & static_cast<ModeFlags>(__llvm_libc::File::OpenMode::APPEND))
      write_append = true;
  }

  void init(char *buffer, size_t buflen, int bufmode, bool owned,
            ModeFlags modeflags) {
    File::init(this, &str_write, &str_read, &str_seek, &str_close, &str_flush,
               buffer, buflen, bufmode, owned, modeflags);
    pos = eof_marker = 0;
    if (modeflags & static_cast<ModeFlags>(__llvm_libc::File::OpenMode::APPEND))
      write_append = true;
    else
      write_append = false;
  }

  void reset() { pos = 0; }
  size_t get_pos() const { return pos; }
  char *get_str() { return str; }

  // Use this method to prefill the file.
  void reset_and_fill(const char *data, size_t len) {
    size_t i;
    for (i = 0; i < len && i < SIZE; ++i) {
      str[i] = data[i];
    }
    pos = 0;
    eof_marker = i;
  }
};

size_t StringFile::str_read(__llvm_libc::File *f, void *data, size_t len) {
  StringFile *sf = static_cast<StringFile *>(f);
  if (sf->pos >= sf->eof_marker)
    return 0;
  size_t i = 0;
  for (i = 0; i < len; ++i)
    reinterpret_cast<char *>(data)[i] = sf->str[sf->pos + i];
  sf->pos += i;
  return i;
}

size_t StringFile::str_write(__llvm_libc::File *f, const void *data,
                             size_t len) {
  StringFile *sf = static_cast<StringFile *>(f);
  if (sf->write_append)
    sf->pos = sf->eof_marker;
  if (sf->pos >= SIZE)
    return 0;
  size_t i = 0;
  for (i = 0; i < len && sf->pos < SIZE; ++i, ++sf->pos)
    sf->str[sf->pos] = reinterpret_cast<const char *>(data)[i];
  // Move the eof marker if the data was written beyond the current eof marker.
  if (sf->pos > sf->eof_marker)
    sf->eof_marker = sf->pos;
  return i;
}

int StringFile::str_seek(__llvm_libc::File *f, long offset, int whence) {
  StringFile *sf = static_cast<StringFile *>(f);
  if (whence == SEEK_SET)
    sf->pos = offset;
  if (whence == SEEK_CUR)
    sf->pos += offset;
  if (whence == SEEK_END)
    sf->pos = SIZE + offset;
  return 0;
}

StringFile *new_string_file(char *buffer, size_t buflen, int bufmode,
                            bool owned, const char *mode) {
  StringFile *f = reinterpret_cast<StringFile *>(malloc(sizeof(StringFile)));
  f->init(buffer, buflen, bufmode, owned, __llvm_libc::File::mode_flags(mode));
  return f;
}

TEST(LlvmLibcFileTest, WriteOnly) {
  const char data[] = "hello, file";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(data) * 3 / 2;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f = new_string_file(file_buffer, FILE_BUFFER_SIZE, 0, false, "w");

  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)));
  EXPECT_EQ(f->get_pos(), size_t(0)); // Data is buffered in the file stream
  ASSERT_EQ(f->flush(), 0);
  EXPECT_EQ(f->get_pos(), sizeof(data)); // Data should now be available
  EXPECT_STREQ(f->get_str(), data);

  f->reset();
  ASSERT_EQ(f->get_pos(), size_t(0));
  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)));
  EXPECT_EQ(f->get_pos(), size_t(0)); // Data is buffered in the file stream
  // The second write should trigger a buffer flush.
  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)));
  EXPECT_GE(f->get_pos(), size_t(0));
  ASSERT_EQ(f->flush(), 0);
  EXPECT_EQ(f->get_pos(), 2 * sizeof(data));

  char read_data[sizeof(data)];
  // This is not a readable file.
  EXPECT_EQ(f->read(read_data, sizeof(data)), size_t(0));
  EXPECT_TRUE(f->error());
  EXPECT_NE(errno, 0);
  errno = 0;

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, ReadOnly) {
  const char initial_content[] = "1234567890987654321";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(initial_content);
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f = new_string_file(file_buffer, FILE_BUFFER_SIZE, 0, false, "r");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  constexpr size_t READ_SIZE = sizeof(initial_content) / 2;
  char read_data[READ_SIZE];
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE));
  EXPECT_FALSE(f->iseof());
  // Reading less than file buffer worth will still read one
  // full buffer worth of data.
  EXPECT_STREQ(file_buffer, initial_content);
  EXPECT_STREQ(file_buffer, f->get_str());
  EXPECT_EQ(FILE_BUFFER_SIZE, f->get_pos());
  // The read data should match what was supposed to be read anyway.
  MemoryView src1(initial_content, READ_SIZE), dst1(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src1, dst1);

  // Reading another buffer worth should read out everything in
  // the file.
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE));
  EXPECT_FALSE(f->iseof());
  MemoryView src2(initial_content + READ_SIZE, READ_SIZE),
      dst2(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src2, dst2);

  // Another read should trigger an EOF.
  ASSERT_GT(READ_SIZE, f->read(read_data, READ_SIZE));
  EXPECT_TRUE(f->iseof());

  // Reset the pos to the beginning of the file which should allow
  // reading again.
  for (size_t i = 0; i < READ_SIZE; ++i)
    read_data[i] = 0;
  f->seek(0, SEEK_SET);
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE));
  MemoryView src3(initial_content, READ_SIZE), dst3(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src3, dst3);

  // This is not a writable file.
  EXPECT_EQ(f->write(initial_content, sizeof(initial_content)), size_t(0));
  EXPECT_TRUE(f->error());
  EXPECT_NE(errno, 0);
  errno = 0;

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, ReadSeekCurAndRead) {
  const char initial_content[] = "1234567890987654321";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(initial_content);
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f = new_string_file(file_buffer, FILE_BUFFER_SIZE, 0, false, "r");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  constexpr size_t READ_SIZE = 5;
  char data[READ_SIZE];
  data[READ_SIZE - 1] = '\0';
  ASSERT_EQ(f->read(data, READ_SIZE - 1), READ_SIZE - 1);
  ASSERT_STREQ(data, "1234");
  ASSERT_EQ(f->seek(5, SEEK_CUR), 0);
  ASSERT_EQ(f->read(data, READ_SIZE - 1), READ_SIZE - 1);
  ASSERT_STREQ(data, "0987");
  ASSERT_EQ(f->seek(-5, SEEK_CUR), 0);
  ASSERT_EQ(f->read(data, READ_SIZE - 1), READ_SIZE - 1);
  ASSERT_STREQ(data, "9098");
  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, AppendOnly) {
  const char initial_content[] = "1234567890987654321";
  const char write_data[] = "append";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(write_data) * 3 / 2;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f = new_string_file(file_buffer, FILE_BUFFER_SIZE, 0, false, "a");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  constexpr size_t READ_SIZE = 5;
  char read_data[READ_SIZE];
  // This is not a readable file.
  ASSERT_EQ(f->read(read_data, READ_SIZE), size_t(0));
  EXPECT_TRUE(f->error());
  EXPECT_NE(errno, 0);
  errno = 0;

  // Write should succeed but will be buffered in the file stream.
  ASSERT_EQ(f->write(write_data, sizeof(write_data)), sizeof(write_data));
  EXPECT_EQ(f->get_pos(), size_t(0));
  // Flushing will write to the file.
  EXPECT_EQ(f->flush(), int(0));
  EXPECT_EQ(f->get_pos(), sizeof(write_data) + sizeof(initial_content));

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, WriteUpdate) {
  const char data[] = "hello, file";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(data) * 3 / 2;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, 0, false, "w+");

  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)));
  EXPECT_EQ(f->get_pos(), size_t(0)); // Data is buffered in the file stream

  ASSERT_EQ(f->seek(0, SEEK_SET), 0);

  // Seek flushes the stream buffer so we can read the previously written data.
  char read_data[sizeof(data)];
  ASSERT_EQ(f->read(read_data, sizeof(data)), sizeof(data));
  EXPECT_STREQ(read_data, data);

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, ReadUpdate) {
  const char initial_content[] = "1234567890987654321";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(initial_content);
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, 0, false, "r+");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  constexpr size_t READ_SIZE = sizeof(initial_content) / 2;
  char read_data[READ_SIZE];
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE));
  EXPECT_FALSE(f->iseof());
  // Reading less than file buffer worth will still read one
  // full buffer worth of data.
  EXPECT_STREQ(file_buffer, initial_content);
  EXPECT_STREQ(file_buffer, f->get_str());
  EXPECT_EQ(FILE_BUFFER_SIZE, f->get_pos());
  // The read data should match what was supposed to be read anyway.
  MemoryView src1(initial_content, READ_SIZE), dst1(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src1, dst1);

  ASSERT_EQ(f->seek(0, SEEK_SET), 0);
  const char write_data[] = "hello, file";
  ASSERT_EQ(sizeof(write_data), f->write(write_data, sizeof(write_data)));
  EXPECT_STREQ(file_buffer, write_data);
  ASSERT_EQ(f->flush(), 0);
  MemoryView dst2(f->get_str(), sizeof(write_data)),
      src2(write_data, sizeof(write_data));
  EXPECT_MEM_EQ(src2, dst2);

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, AppendUpdate) {
  const char initial_content[] = "1234567890987654321";
  const char data[] = "hello, file";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(data) * 3 / 2;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, 0, false, "a+");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)));
  EXPECT_EQ(f->get_pos(), size_t(0)); // Data is buffered in the file stream
  ASSERT_EQ(f->flush(), 0);
  // The flush should write |data| to the endof the file.
  EXPECT_EQ(f->get_pos(), sizeof(data) + sizeof(initial_content));

  ASSERT_EQ(f->seek(0, SEEK_SET), 0);
  // Seeking to the beginning of the file should not affect the place
  // where write happens.
  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)));
  ASSERT_EQ(f->flush(), 0);
  EXPECT_EQ(f->get_pos(), sizeof(data) * 2 + sizeof(initial_content));
  MemoryView src1(initial_content, sizeof(initial_content)),
      dst1(f->get_str(), sizeof(initial_content));
  EXPECT_MEM_EQ(src1, dst1);
  MemoryView src2(data, sizeof(data)),
      dst2(f->get_str() + sizeof(initial_content), sizeof(data));
  EXPECT_MEM_EQ(src2, dst2);
  MemoryView src3(data, sizeof(data)),
      dst3(f->get_str() + sizeof(initial_content) + sizeof(data), sizeof(data));
  EXPECT_MEM_EQ(src3, dst3);

  // Reads can happen from any point.
  ASSERT_EQ(f->seek(0, SEEK_SET), 0);
  constexpr size_t READ_SIZE = 10;
  char read_data[READ_SIZE];
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE));
  MemoryView src4(initial_content, READ_SIZE), dst4(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src4, dst4);

  ASSERT_EQ(f->close(), 0);
}
