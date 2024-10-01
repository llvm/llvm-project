//===-- Unittests for platform independent file class ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/new.h"
#include "src/__support/File/file.h"
#include "src/__support/error_or.h"
#include "test/UnitTest/MemoryMatcher.h"
#include "test/UnitTest/Test.h"

#include <stdlib.h>

using ModeFlags = LIBC_NAMESPACE::File::ModeFlags;
using MemoryView = LIBC_NAMESPACE::testing::MemoryView;
using LIBC_NAMESPACE::ErrorOr;
using LIBC_NAMESPACE::File;
using LIBC_NAMESPACE::FileIOResult;

class StringFile : public File {
  static constexpr size_t SIZE = 512;
  size_t pos;
  char str[SIZE] = {0};
  size_t eof_marker;
  bool write_append;

  static FileIOResult str_read(LIBC_NAMESPACE::File *f, void *data, size_t len);
  static FileIOResult str_write(LIBC_NAMESPACE::File *f, const void *data,
                                size_t len);
  static ErrorOr<off_t> str_seek(LIBC_NAMESPACE::File *f, off_t offset,
                                 int whence);
  static int str_close(LIBC_NAMESPACE::File *f) {
    delete reinterpret_cast<StringFile *>(f);
    return 0;
  }

public:
  explicit StringFile(char *buffer, size_t buflen, int bufmode, bool owned,
                      ModeFlags modeflags)
      : LIBC_NAMESPACE::File(&str_write, &str_read, &str_seek, &str_close,
                             reinterpret_cast<uint8_t *>(buffer), buflen,
                             bufmode, owned, modeflags),
        pos(0), eof_marker(0), write_append(false) {
    if (modeflags &
        static_cast<ModeFlags>(LIBC_NAMESPACE::File::OpenMode::APPEND))
      write_append = true;
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

FileIOResult StringFile::str_read(LIBC_NAMESPACE::File *f, void *data,
                                  size_t len) {
  StringFile *sf = static_cast<StringFile *>(f);
  if (sf->pos >= sf->eof_marker)
    return 0;
  size_t i = 0;
  for (i = 0; i < len; ++i)
    reinterpret_cast<char *>(data)[i] = sf->str[sf->pos + i];
  sf->pos += i;
  return i;
}

FileIOResult StringFile::str_write(LIBC_NAMESPACE::File *f, const void *data,
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

ErrorOr<off_t> StringFile::str_seek(LIBC_NAMESPACE::File *f, off_t offset,
                                    int whence) {
  StringFile *sf = static_cast<StringFile *>(f);
  if (whence == SEEK_SET)
    sf->pos = offset;
  if (whence == SEEK_CUR)
    sf->pos += offset;
  if (whence == SEEK_END)
    sf->pos = SIZE + offset;
  return sf->pos;
}

StringFile *new_string_file(char *buffer, size_t buflen, int bufmode,
                            bool owned, const char *mode) {
  LIBC_NAMESPACE::AllocChecker ac;
  // We will just assume the allocation succeeds. We cannot test anything
  // otherwise.
  return new (ac) StringFile(buffer, buflen, bufmode, owned,
                             LIBC_NAMESPACE::File::mode_flags(mode));
}

TEST(LlvmLibcFileTest, WriteOnly) {
  const char data[] = "hello, file";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(data) * 3 / 2;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w");

  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)).value);
  EXPECT_EQ(f->get_pos(), size_t(0)); // Data is buffered in the file stream
  ASSERT_EQ(f->flush(), 0);
  EXPECT_EQ(f->get_pos(), sizeof(data)); // Data should now be available
  EXPECT_STREQ(f->get_str(), data);

  f->reset();
  ASSERT_EQ(f->get_pos(), size_t(0));
  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)).value);
  EXPECT_EQ(f->get_pos(), size_t(0)); // Data is buffered in the file stream
  // The second write should trigger a buffer flush.
  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)).value);
  EXPECT_GE(f->get_pos(), size_t(0));
  ASSERT_EQ(f->flush(), 0);
  EXPECT_EQ(f->get_pos(), 2 * sizeof(data));
  MemoryView src1("hello, file\0hello, file", sizeof(data) * 2),
      dst1(f->get_str(), sizeof(data) * 2);
  EXPECT_MEM_EQ(src1, dst1);

  char read_data[sizeof(data)];
  {
    // This is not a readable file.
    auto result = f->read(read_data, sizeof(data));
    EXPECT_EQ(result.value, size_t(0));
    EXPECT_TRUE(f->error());
    EXPECT_TRUE(result.has_error());
  }

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, WriteLineBuffered) {
  const char data[] = "hello\n file";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(data) * 3 / 2;

  char file_buffer_line[FILE_BUFFER_SIZE];
  char file_buffer_full[FILE_BUFFER_SIZE];

  StringFile *f_line =
      new_string_file(file_buffer_line, FILE_BUFFER_SIZE, _IOLBF, false, "w");
  // We also initialize a fully buffered file we'll do the same writes to for
  // comparison.
  StringFile *f_full =
      new_string_file(file_buffer_full, FILE_BUFFER_SIZE, _IOFBF, false, "w");

  ASSERT_EQ(sizeof(data), f_line->write(data, sizeof(data)).value);
  ASSERT_EQ(sizeof(data), f_full->write(data, sizeof(data)).value);

  EXPECT_EQ(f_line->get_pos(), size_t(6)); // buffer after the newline
  EXPECT_EQ(f_full->get_pos(), size_t(0)); // buffer all of data

  MemoryView src1("hello\n", 6), dst1(f_line->get_str(), 6);
  EXPECT_MEM_EQ(src1, dst1);

  // We can't check the data in f_full, since no data has been written.

  const char data2[] = "longer for an \n overflow";

  ASSERT_EQ(sizeof(data2), f_line->write(data2, sizeof(data2)).value);
  // The line buffer's initial contents should be " file\0"
  // Writing data2 should write up until the newline, even though that doesn't
  // all fit in the buffer.

  ASSERT_EQ(sizeof(data2), f_full->write(data2, sizeof(data2)).value);
  // The full buffer's initial contents should be "hello\n file\0"
  // Writing data2 should cause a flush of the buffer, as well as the remainder
  // to be written directly since it doesn't fit in the buffer.

  EXPECT_EQ(f_line->get_pos(), size_t(27));
  EXPECT_EQ(f_full->get_pos(), sizeof(data) + sizeof(data2));

  MemoryView src2("hello\n file\0longer for an \n", 27),
      dst2(f_line->get_str(), 27);
  EXPECT_MEM_EQ(src2, dst2);

  MemoryView src3("hello\n file\0longer for an \n overflow", 37),
      dst_full_final(f_full->get_str(), 37);
  EXPECT_MEM_EQ(src3, dst_full_final);

  ASSERT_EQ(f_line->flush(), 0);
  ASSERT_EQ(f_full->flush(), 0);

  EXPECT_EQ(f_line->get_pos(), sizeof(data) + sizeof(data2));
  MemoryView dst_line_final(f_line->get_str(), 37);
  EXPECT_MEM_EQ(src3, dst_line_final);
  EXPECT_MEM_EQ(src3, dst_full_final);

  ASSERT_EQ(f_line->close(), 0);
  ASSERT_EQ(f_full->close(), 0);
}

TEST(LlvmLibcFileTest, WriteUnbuffered) {
  const char data[] = "written immediately";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(data) + 1;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IONBF, false, "w");

  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)).value);
  EXPECT_EQ(f->get_pos(),
            sizeof(data)); // no buffering means this is written immediately.
  EXPECT_STREQ(f->get_str(), data);

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, ReadOnly) {
  const char initial_content[] = "1234567890987654321";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(initial_content);
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "r");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  constexpr size_t READ_SIZE = sizeof(initial_content) / 2;
  char read_data[READ_SIZE];
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE).value);
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
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE).value);
  EXPECT_FALSE(f->iseof());
  MemoryView src2(initial_content + READ_SIZE, READ_SIZE),
      dst2(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src2, dst2);

  // Another read should trigger an EOF.
  ASSERT_GT(READ_SIZE, f->read(read_data, READ_SIZE).value);
  EXPECT_TRUE(f->iseof());

  // Reset the pos to the beginning of the file which should allow
  // reading again.
  for (size_t i = 0; i < READ_SIZE; ++i)
    read_data[i] = 0;
  f->seek(0, SEEK_SET);
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE).value);
  MemoryView src3(initial_content, READ_SIZE), dst3(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src3, dst3);

  {
    // This is not a writable file.
    auto result = f->write(initial_content, sizeof(initial_content));
    EXPECT_EQ(result.value, size_t(0));
    EXPECT_TRUE(f->error());
    EXPECT_TRUE(result.has_error());
  }

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, ReadSeekCurAndRead) {
  const char initial_content[] = "1234567890987654321";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(initial_content);
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "r");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  constexpr size_t READ_SIZE = 5;
  char data[READ_SIZE];
  data[READ_SIZE - 1] = '\0';
  ASSERT_EQ(f->read(data, READ_SIZE - 1).value, READ_SIZE - 1);
  ASSERT_STREQ(data, "1234");
  ASSERT_EQ(f->seek(5, SEEK_CUR).value(), 0);
  ASSERT_EQ(f->read(data, READ_SIZE - 1).value, READ_SIZE - 1);
  ASSERT_STREQ(data, "0987");
  ASSERT_EQ(f->seek(-5, SEEK_CUR).value(), 0);
  ASSERT_EQ(f->read(data, READ_SIZE - 1).value, READ_SIZE - 1);
  ASSERT_STREQ(data, "9098");
  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, AppendOnly) {
  const char initial_content[] = "1234567890987654321";
  const char write_data[] = "append";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(write_data) * 3 / 2;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "a");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  constexpr size_t READ_SIZE = 5;
  char read_data[READ_SIZE];

  {
    // This is not a readable file.
    auto result = f->read(read_data, READ_SIZE);
    EXPECT_EQ(result.value, size_t(0));
    EXPECT_TRUE(f->error());
    EXPECT_TRUE(result.has_error());
  }

  // Write should succeed but will be buffered in the file stream.
  ASSERT_EQ(f->write(write_data, sizeof(write_data)).value, sizeof(write_data));
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
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w+");

  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)).value);
  EXPECT_EQ(f->get_pos(), size_t(0)); // Data is buffered in the file stream

  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);

  // Seek flushes the stream buffer so we can read the previously written data.
  char read_data[sizeof(data)];
  ASSERT_EQ(f->read(read_data, sizeof(data)).value, sizeof(data));
  EXPECT_STREQ(read_data, data);

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, ReadUpdate) {
  const char initial_content[] = "1234567890987654321";
  constexpr size_t FILE_BUFFER_SIZE = sizeof(initial_content);
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "r+");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  constexpr size_t READ_SIZE = sizeof(initial_content) / 2;
  char read_data[READ_SIZE];
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE).value);
  EXPECT_FALSE(f->iseof());
  // Reading less than file buffer worth will still read one
  // full buffer worth of data.
  EXPECT_STREQ(file_buffer, initial_content);
  EXPECT_STREQ(file_buffer, f->get_str());
  EXPECT_EQ(FILE_BUFFER_SIZE, f->get_pos());
  // The read data should match what was supposed to be read anyway.
  MemoryView src1(initial_content, READ_SIZE), dst1(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src1, dst1);

  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);
  const char write_data[] = "hello, file";
  ASSERT_EQ(sizeof(write_data), f->write(write_data, sizeof(write_data)).value);
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
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "a+");
  f->reset_and_fill(initial_content, sizeof(initial_content));

  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)).value);
  EXPECT_EQ(f->get_pos(), size_t(0)); // Data is buffered in the file stream
  ASSERT_EQ(f->flush(), 0);
  // The flush should write |data| to the endof the file.
  EXPECT_EQ(f->get_pos(), sizeof(data) + sizeof(initial_content));

  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);
  // Seeking to the beginning of the file should not affect the place
  // where write happens.
  ASSERT_EQ(sizeof(data), f->write(data, sizeof(data)).value);
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
  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);
  constexpr size_t READ_SIZE = 10;
  char read_data[READ_SIZE];
  ASSERT_EQ(READ_SIZE, f->read(read_data, READ_SIZE).value);
  MemoryView src4(initial_content, READ_SIZE), dst4(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src4, dst4);

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, SmallBuffer) {
  const char WRITE_DATA[] = "small buffer";
  constexpr size_t WRITE_SIZE = sizeof(WRITE_DATA);
  constexpr size_t FILE_BUFFER_SIZE = sizeof(WRITE_DATA) / 2 - 1;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w");

  ASSERT_EQ(WRITE_SIZE, f->write(WRITE_DATA, WRITE_SIZE).value);
  // Since data much larger than the buffer is being written, all of it should
  // be available in the file without a flush operation.
  EXPECT_EQ(f->get_pos(), sizeof(WRITE_DATA));
  ASSERT_STREQ(f->get_str(), WRITE_DATA);

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, ZeroLengthBuffer) {
  const char WRITE_DATA[] = "small buffer";
  constexpr size_t WRITE_SIZE = sizeof(WRITE_DATA);
  StringFile *f_fbf = new_string_file(nullptr, 0, _IOFBF, true, "w");
  StringFile *f_lbf = new_string_file(nullptr, 0, _IOLBF, true, "w");
  StringFile *f_nbf = new_string_file(nullptr, 0, _IONBF, true, "w");

  ASSERT_EQ(WRITE_SIZE, f_fbf->write(WRITE_DATA, WRITE_SIZE).value);
  ASSERT_EQ(WRITE_SIZE, f_lbf->write(WRITE_DATA, WRITE_SIZE).value);
  ASSERT_EQ(WRITE_SIZE, f_nbf->write(WRITE_DATA, WRITE_SIZE).value);
  // Since there is no buffer space, all of the written data should
  // be available in the file without a flush operation.
  EXPECT_EQ(f_fbf->get_pos(), sizeof(WRITE_DATA));
  EXPECT_EQ(f_lbf->get_pos(), sizeof(WRITE_DATA));
  EXPECT_EQ(f_nbf->get_pos(), sizeof(WRITE_DATA));
  ASSERT_STREQ(f_fbf->get_str(), WRITE_DATA);
  ASSERT_STREQ(f_lbf->get_str(), WRITE_DATA);
  ASSERT_STREQ(f_nbf->get_str(), WRITE_DATA);

  ASSERT_EQ(f_fbf->close(), 0);
  ASSERT_EQ(f_lbf->close(), 0);
  ASSERT_EQ(f_nbf->close(), 0);
}

TEST(LlvmLibcFileTest, WriteNothing) {
  const char WRITE_DATA[] = "";
  constexpr size_t WRITE_SIZE = 0;
  constexpr size_t FILE_BUFFER_SIZE = 5;
  char file_buffer_fbf[FILE_BUFFER_SIZE];
  char file_buffer_lbf[FILE_BUFFER_SIZE];
  char file_buffer_nbf[FILE_BUFFER_SIZE];
  StringFile *f_fbf =
      new_string_file(file_buffer_fbf, FILE_BUFFER_SIZE, _IOFBF, false, "w");
  StringFile *f_lbf =
      new_string_file(file_buffer_lbf, FILE_BUFFER_SIZE, _IOLBF, false, "w");
  StringFile *f_nbf =
      new_string_file(file_buffer_nbf, FILE_BUFFER_SIZE, _IONBF, false, "w");

  ASSERT_EQ(WRITE_SIZE, f_fbf->write(WRITE_DATA, WRITE_SIZE).value);
  ASSERT_EQ(WRITE_SIZE, f_lbf->write(WRITE_DATA, WRITE_SIZE).value);
  ASSERT_EQ(WRITE_SIZE, f_nbf->write(WRITE_DATA, WRITE_SIZE).value);

  ASSERT_FALSE(f_fbf->error_unlocked());
  ASSERT_FALSE(f_lbf->error_unlocked());
  ASSERT_FALSE(f_nbf->error_unlocked());

  ASSERT_EQ(f_fbf->close(), 0);
  ASSERT_EQ(f_lbf->close(), 0);
  ASSERT_EQ(f_nbf->close(), 0);
}
