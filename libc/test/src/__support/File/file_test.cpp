//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for platform independent file class.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/off_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/wchar_macros.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/File/file.h"
#include "src/__support/File/file_impl.h"
#include "src/__support/File/file_mode.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/MemoryMatcher.h"
#include "test/UnitTest/Test.h"

using MemoryView = LIBC_NAMESPACE::testing::MemoryView;
using LIBC_NAMESPACE::CookieOps;
using LIBC_NAMESPACE::Error;
using LIBC_NAMESPACE::ErrorOr;
using LIBC_NAMESPACE::FileIOResult;
using LIBC_NAMESPACE::FileMode;
using LIBC_NAMESPACE::cpp::scope_exit;

struct TestFilePlatform;
using TestFile = LIBC_NAMESPACE::FileImpl<TestFilePlatform>;
using TestCustomFile = LIBC_NAMESPACE::CustomFileImpl<TestFile>;
using TestCookieFile = LIBC_NAMESPACE::CookieFileImpl<TestFile>;

struct TestFilePlatform {
  using StreamBase = TestFile;

  static FileIOResult write(TestFile *f, const void *data, size_t len);
  static FileIOResult read(TestFile *f, void *data, size_t len);
  static ErrorOr<off_t> seek(TestFile *f, off_t offset, int whence);
  static int close(TestFile *f);
};

static_assert(sizeof(TestCustomFile) ==
              sizeof(TestFile) + sizeof(TestFile::Ops));
static_assert(sizeof(TestCookieFile) ==
              sizeof(TestFile) + sizeof(void *) + sizeof(CookieOps));

class StaticBufferFile : public TestFile {
  static constexpr size_t SIZE = 512;
  size_t pos = 0;
  char str[SIZE] = {0};
  size_t eof_marker = 0;
  bool write_append = false;
  size_t *write_calls = nullptr;
  size_t *read_calls = nullptr;
  size_t *seek_calls = nullptr;
  size_t *close_calls = nullptr;

  friend struct TestFilePlatform;

public:
  StaticBufferFile(char *buffer, size_t buflen, int bufmode, bool owned,
                   FileMode mode, size_t *writes = nullptr,
                   size_t *reads = nullptr, size_t *seeks = nullptr,
                   size_t *closes = nullptr)
      : TestFile(reinterpret_cast<uint8_t *>(buffer), buflen, bufmode, owned,
                 mode),
        write_append(mode.is_append()), write_calls(writes), read_calls(reads),
        seek_calls(seeks), close_calls(closes) {}

  size_t get_pos() const { return pos; }
  char *get_str() { return str; }

  void reset_and_fill(const char *data, size_t len) {
    size_t i = 0;
    for (; i < len && i < SIZE; ++i)
      str[i] = data[i];
    pos = 0;
    eof_marker = i;
  }
};

FileIOResult TestFilePlatform::write(TestFile *f, const void *data,
                                     size_t len) {
  auto *sf = static_cast<StaticBufferFile *>(f);
  if (sf->write_calls != nullptr)
    ++(*sf->write_calls);
  if (sf->write_append)
    sf->pos = sf->eof_marker;
  if (sf->pos >= StaticBufferFile::SIZE)
    return 0;
  size_t i = 0;
  for (; i < len && sf->pos < StaticBufferFile::SIZE; ++i, ++sf->pos)
    sf->str[sf->pos] = reinterpret_cast<const char *>(data)[i];
  if (sf->pos > sf->eof_marker)
    sf->eof_marker = sf->pos;
  return i;
}

FileIOResult TestFilePlatform::read(TestFile *f, void *data, size_t len) {
  auto *sf = static_cast<StaticBufferFile *>(f);
  if (sf->read_calls != nullptr)
    ++(*sf->read_calls);
  if (sf->pos >= sf->eof_marker)
    return 0;
  size_t avail = sf->eof_marker - sf->pos;
  size_t to_copy = len < avail ? len : avail;
  for (size_t i = 0; i < to_copy; ++i)
    reinterpret_cast<char *>(data)[i] = sf->str[sf->pos + i];
  sf->pos += to_copy;
  return to_copy;
}

ErrorOr<off_t> TestFilePlatform::seek(TestFile *f, off_t offset, int whence) {
  auto *sf = static_cast<StaticBufferFile *>(f);
  if (sf->seek_calls != nullptr)
    ++(*sf->seek_calls);
  if (whence == SEEK_SET)
    sf->pos = static_cast<size_t>(offset);
  else if (whence == SEEK_CUR)
    sf->pos = static_cast<size_t>(static_cast<off_t>(sf->pos) + offset);
  else if (whence == SEEK_END)
    sf->pos = static_cast<size_t>(sf->eof_marker + offset);
  else
    return Error(EINVAL);
  return static_cast<off_t>(sf->pos);
}

int TestFilePlatform::close(TestFile *f) {
  auto *sf = static_cast<StaticBufferFile *>(f);
  if (sf->close_calls != nullptr)
    ++(*sf->close_calls);
  delete sf;
  return 0;
}

class StringFile : public TestCustomFile {
  static constexpr size_t SIZE = 512;
  size_t pos;
  char str[SIZE] = {0};
  size_t eof_marker;
  bool write_append;

  static FileIOResult str_read(TestFile *f, void *data, size_t len);
  static FileIOResult str_write(TestFile *f, const void *data, size_t len);
  static ErrorOr<off_t> str_seek(TestFile *f, off_t offset, int whence);
  static int str_close(TestFile *f) {
    delete static_cast<StringFile *>(f);
    return 0;
  }

public:
  explicit StringFile(char *buffer, size_t buflen, int bufmode, bool owned,
                      FileMode mode)
      : TestCustomFile({&str_write, &str_read, &str_seek, &str_close},
                       reinterpret_cast<uint8_t *>(buffer), buflen, bufmode,
                       owned, mode),
        pos(0), eof_marker(0), write_append(false) {
    if (mode.is_append())
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

FileIOResult StringFile::str_read(TestFile *f, void *data, size_t len) {
  StringFile *sf = static_cast<StringFile *>(f);
  if (sf->pos >= sf->eof_marker)
    return 0;
  size_t i = 0;
  for (i = 0; i < len; ++i)
    reinterpret_cast<char *>(data)[i] = sf->str[sf->pos + i];
  sf->pos += i;
  return i;
}

FileIOResult StringFile::str_write(TestFile *f, const void *data, size_t len) {
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

ErrorOr<off_t> StringFile::str_seek(TestFile *f, off_t offset, int whence) {
  StringFile *sf = static_cast<StringFile *>(f);
  if (whence == SEEK_SET)
    sf->pos = static_cast<size_t>(offset);
  else if (whence == SEEK_CUR)
    sf->pos = static_cast<size_t>(static_cast<off_t>(sf->pos) + offset);
  else if (whence == SEEK_END)
    sf->pos = static_cast<size_t>(SIZE + offset);
  return sf->pos;
}

StringFile *new_string_file(char *buffer, size_t buflen, int bufmode,
                            bool owned, const char *mode) {
  LIBC_NAMESPACE::AllocChecker ac;
  const FileMode file_mode(mode);
  // We will just assume the allocation succeeds. We cannot test anything
  // otherwise.
  return new (ac) StringFile(buffer, buflen, bufmode, owned, file_mode);
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

TEST(LlvmLibcFileTest, WriteSplit) {
  constexpr size_t FILE_BUFFER_SIZE = 8;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w");

  static constexpr size_t AVAIL = 12;
  f->seek(-static_cast<off_t>(AVAIL), SEEK_END);

  const char data[] = "hello";
  ASSERT_EQ(sizeof(data) - 1, f->write(data, sizeof(data) - 1).value);

  const char data2[] = " extra data";
  static constexpr size_t WR_EXPECTED = AVAIL - (sizeof(data) - 1);
  ASSERT_EQ(WR_EXPECTED, f->write(data2, sizeof(data2) - 1).value);
  EXPECT_TRUE(f->error());
  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, WideCharOrientation) {
  constexpr size_t FILE_BUFFER_SIZE = 512;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w+");

  f->write(L"A", 1);

  auto write_res = f->write("B", 1);
  EXPECT_EQ(write_res.value, size_t(0));
  EXPECT_TRUE(f->error());

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, ByteCharOrientation) {
  constexpr size_t FILE_BUFFER_SIZE = 512;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w+");

  f->write("A", 1);

  auto write_res = f->write(L"B", 1);
  EXPECT_EQ(write_res.value, size_t(0));
  EXPECT_TRUE(f->error());

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, Ungetwc) {
  constexpr size_t FILE_BUFFER_SIZE = 512;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w+");

  f->write(L"A", 1);
  f->flush();
  f->seek(0, SEEK_SET);

  wchar_t ws_out[2];
  auto read_res = f->read(ws_out, 1);
  ASSERT_EQ(read_res.value, size_t(1));
  EXPECT_EQ(static_cast<uint32_t>(ws_out[0]), static_cast<uint32_t>(L'A'));

  auto unget_res = f->ungetwc(L'B');
  ASSERT_TRUE(unget_res.has_value());
  EXPECT_EQ(static_cast<uint32_t>(unget_res.value()),
            static_cast<uint32_t>(L'B'));
  auto read_res2 = f->read(ws_out, 1);

  ASSERT_EQ(read_res2.value, size_t(1));
  EXPECT_EQ(static_cast<uint32_t>(ws_out[0]), static_cast<uint32_t>(L'B'));

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, WideStringIO) {
  constexpr size_t FILE_BUFFER_SIZE = 100;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w+");
  ASSERT_FALSE(f == nullptr);

  const wchar_t *ws = L"Hello, World!";
  size_t len = 13;

  auto write_res = f->write(ws, len);
  ASSERT_FALSE(write_res.has_error());
  EXPECT_EQ(write_res.value, len);

  ASSERT_EQ(f->flush(), 0); // Ensure everything is written to StringFile

  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);

  wchar_t read_buf[20];
  auto read_res = f->read(read_buf, len);
  ASSERT_FALSE(read_res.has_error());
  EXPECT_EQ(read_res.value, len);

  for (size_t i = 0; i < len; ++i) {
    EXPECT_EQ(static_cast<uint32_t>(read_buf[i]), static_cast<uint32_t>(ws[i]));
  }

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, TrySetOrientation) {
  constexpr size_t FILE_BUFFER_SIZE = 100;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "r+");
  ASSERT_FALSE(f == nullptr);

  EXPECT_EQ(static_cast<uint32_t>(f->get_orientation()),
            static_cast<uint32_t>(TestFile::Orientation::UNORIENTED));

  EXPECT_EQ(static_cast<uint32_t>(
                f->try_set_orientation(TestFile::Orientation::WIDE)),
            static_cast<uint32_t>(TestFile::Orientation::WIDE));
  EXPECT_EQ(static_cast<uint32_t>(f->get_orientation()),
            static_cast<uint32_t>(TestFile::Orientation::WIDE));

  EXPECT_EQ(
      static_cast<uint32_t>(
          f->try_set_orientation(TestFile::Orientation::BYTE)),
      static_cast<uint32_t>(TestFile::Orientation::WIDE)); // Cannot change
  EXPECT_EQ(static_cast<uint32_t>(f->get_orientation()),
            static_cast<uint32_t>(TestFile::Orientation::WIDE));

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, UngetwcMultiByte) {
  constexpr size_t FILE_BUFFER_SIZE = 512;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w+");

  f->write(L"€", 1);
  f->flush();
  f->seek(0, SEEK_SET);

  wchar_t ws_out[2];
  auto read_res = f->read(ws_out, 1);
  ASSERT_EQ(read_res.value, size_t(1));
  EXPECT_EQ(static_cast<uint32_t>(ws_out[0]), static_cast<uint32_t>(L'€'));

  auto unget_res = f->ungetwc(L'¢');
  ASSERT_TRUE(unget_res.has_value());
  EXPECT_EQ(static_cast<uint32_t>(unget_res.value()),
            static_cast<uint32_t>(L'¢'));

  auto read_res2 = f->read(ws_out, 1);
  ASSERT_EQ(read_res2.value, size_t(1));
  EXPECT_EQ(static_cast<uint32_t>(ws_out[0]), static_cast<uint32_t>(L'¢'));

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, UngetwcUnbufferedMultiByte) {
  StringFile *f = new_string_file(nullptr, 0, _IONBF, true, "w+");
  ASSERT_FALSE(f == nullptr);

  f->write(L"€", 1);
  f->seek(0, SEEK_SET);

  wchar_t ws_out[2];
  auto read_res = f->read(ws_out, 1);
  ASSERT_EQ(read_res.value, size_t(1));
  EXPECT_EQ(static_cast<uint32_t>(ws_out[0]), static_cast<uint32_t>(L'€'));

  auto unget_res = f->ungetwc(L'¢');
  ASSERT_TRUE(unget_res.has_value());
  EXPECT_EQ(static_cast<uint32_t>(unget_res.value()),
            static_cast<uint32_t>(L'¢'));

  auto read_res2 = f->read(ws_out, 1);
  ASSERT_EQ(read_res2.value, size_t(1));
  EXPECT_EQ(static_cast<uint32_t>(ws_out[0]), static_cast<uint32_t>(L'¢'));

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, WideStringIO_Multibyte) {
  constexpr size_t FILE_BUFFER_SIZE = 100;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w+");
  ASSERT_FALSE(f == nullptr);

  const wchar_t *ws = L"Hello € World!";
  size_t len = 14;

  auto write_res = f->write(ws, len);
  ASSERT_FALSE(write_res.has_error());
  EXPECT_EQ(write_res.value, len);

  ASSERT_EQ(f->flush(), 0);

  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);

  wchar_t read_buf[20];
  auto read_res = f->read(read_buf, len);
  ASSERT_FALSE(read_res.has_error());
  EXPECT_EQ(read_res.value, len);

  for (size_t i = 0; i < len; ++i) {
    EXPECT_EQ(static_cast<uint32_t>(read_buf[i]), static_cast<uint32_t>(ws[i]));
  }

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, SeekResetsMbstate) {
  constexpr size_t FILE_BUFFER_SIZE = 100;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "r+");
  ASSERT_FALSE(f == nullptr);

  f->reset_and_fill("\xE2\x82", 2);

  wchar_t ws_out[1];
  auto read_res = f->read(ws_out, 1);
  EXPECT_EQ(read_res.value, size_t(0));
  EXPECT_TRUE(f->error());

  f->reset_and_fill("A", 1);
  f->seek(0, SEEK_SET);
  f->clearerr();

  auto read_res2 = f->read(ws_out, 1);
  EXPECT_EQ(read_res2.value, size_t(1));
  EXPECT_EQ(static_cast<uint32_t>(ws_out[0]), static_cast<uint32_t>(L'A'));

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, ReadWideNotStopAtNewline) {
  constexpr size_t FILE_BUFFER_SIZE = 100;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "w+");
  ASSERT_FALSE(f == nullptr);

  const wchar_t *ws = L"Hello\nWorld!";
  size_t len = 12;

  auto write_res = f->write(ws, len);
  ASSERT_FALSE(write_res.has_error());
  EXPECT_EQ(write_res.value, len);

  ASSERT_EQ(f->flush(), 0);
  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);

  wchar_t read_buf[20];
  auto read_res = f->read(read_buf, len);
  ASSERT_FALSE(read_res.has_error());
  // Should NOT stop at newline, so should read all 12 characters.
  EXPECT_EQ(read_res.value, len);
  EXPECT_EQ(static_cast<uint32_t>(read_buf[5]), static_cast<uint32_t>(L'\n'));
  EXPECT_EQ(static_cast<uint32_t>(read_buf[11]), static_cast<uint32_t>(L'!'));

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, UngetwcWEOF) {
  constexpr size_t FILE_BUFFER_SIZE = 100;
  char file_buffer[FILE_BUFFER_SIZE];
  StringFile *f =
      new_string_file(file_buffer, FILE_BUFFER_SIZE, _IOFBF, false, "r+");
  ASSERT_FALSE(f == nullptr);

  EXPECT_EQ(static_cast<uint32_t>(f->get_orientation()),
            static_cast<uint32_t>(TestFile::Orientation::UNORIENTED));

  auto unget_res = f->ungetwc(WEOF);
  ASSERT_TRUE(unget_res.has_value());
  EXPECT_EQ(static_cast<uint32_t>(unget_res.value()),
            static_cast<uint32_t>(WEOF));

  EXPECT_EQ(static_cast<uint32_t>(f->get_orientation()),
            static_cast<uint32_t>(TestFile::Orientation::UNORIENTED));

  ASSERT_EQ(f->close(), 0);
}

// A CustomFileImpl subclass with a write callback that simulates short writes.
// This models the behavior of write(2) on pipes, sockets, or FIFOs where
// the kernel may write fewer bytes than requested.
class ShortWriteFile : public TestCustomFile {
  static constexpr size_t SIZE = 512;
  size_t pos;
  char str[SIZE] = {0};
  size_t max_write;

  static FileIOResult short_write(TestFile *f, const void *data, size_t len) {
    ShortWriteFile *sf = static_cast<ShortWriteFile *>(f);
    // Simulate a short write: write at most max_write bytes per call.
    size_t to_write = len < sf->max_write ? len : sf->max_write;
    for (size_t i = 0; i < to_write && sf->pos < SIZE; ++i, ++sf->pos)
      sf->str[sf->pos] = reinterpret_cast<const char *>(data)[i];
    return to_write;
  }

  static FileIOResult short_read(TestFile *f, void *data, size_t len) {
    ShortWriteFile *sf = static_cast<ShortWriteFile *>(f);
    size_t i = 0;
    for (i = 0; i < len && sf->pos < SIZE; ++i)
      reinterpret_cast<char *>(data)[i] = sf->str[sf->pos + i];
    sf->pos += i;
    return i;
  }

  static ErrorOr<off_t> short_seek(TestFile *f, off_t offset, int whence) {
    ShortWriteFile *sf = static_cast<ShortWriteFile *>(f);
    if (whence == SEEK_SET)
      sf->pos = static_cast<size_t>(offset);
    else if (whence == SEEK_CUR)
      sf->pos += offset;
    else if (whence == SEEK_END)
      sf->pos = static_cast<size_t>(SIZE + offset);
    return sf->pos;
  }

  static int short_close(TestFile *f) {
    delete static_cast<ShortWriteFile *>(f);
    return 0;
  }

public:
  explicit ShortWriteFile(char *buffer, size_t buflen, int bufmode, bool owned,
                          FileMode mode, size_t max_write_bytes)
      : TestCustomFile({&short_write, &short_read, &short_seek, &short_close},
                       reinterpret_cast<uint8_t *>(buffer), buflen, bufmode,
                       owned, mode),
        pos(0), max_write(max_write_bytes) {}

  void reset() { pos = 0; }
  size_t get_pos() const { return pos; }
  char *get_str() { return str; }
};

// Verify that a short platform_write of a multi-byte UTF-8 character is
// detected and reported as a failure. POSIX write(2) may perform short
// writes on pipes, sockets, and FIFOs, so a 3-byte character could have
// only 2 bytes accepted by the kernel.
TEST(LlvmLibcFileTest, PartialWideCharWriteDetected) {
  LIBC_NAMESPACE::AllocChecker ac;
  // Unbuffered so writes go directly to platform_write, limited to 2 bytes.
  ShortWriteFile *f =
      new (ac) ShortWriteFile(nullptr, 0, _IONBF, true, FileMode("w"),
                              /*max_write_bytes=*/2);
  ASSERT_FALSE(f == nullptr);

  // € (U+20AC) encodes to 3 UTF-8 bytes: 0xE2 0x82 0xAC.
  // With max_write=2, only 2 of the 3 bytes will be accepted.
  const wchar_t euro = L'€';
  auto result = f->write(&euro, 1);

  // The incomplete character must not be counted as written.
  EXPECT_TRUE(result.has_error());
  EXPECT_EQ(result.value, size_t(0));

  // The error indicator on the stream should be set.
  EXPECT_TRUE(f->error());

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, FileLockRAII) {
  StringFile *f = new_string_file(nullptr, 0, _IONBF, false, "w+");
  ASSERT_FALSE(f == nullptr);

  {
    TestFile::FileLock lock(f);
    ASSERT_EQ(f->write_unlocked("abc", 3).value, size_t(3));
  }

  // After FileLock is destroyed, operations under lock still acquire and
  // release cleanly.
  ASSERT_EQ(f->write("def", 3).value, size_t(3));
  ASSERT_EQ(f->close(), 0);
}

class ShortReadFile : public TestCustomFile {
  const char *src;
  size_t src_len;
  size_t pos;
  size_t max_read;

  static FileIOResult short_write(TestFile *, const void *, size_t) {
    return 0;
  }

  static FileIOResult short_read(TestFile *f, void *data, size_t len) {
    ShortReadFile *sf = static_cast<ShortReadFile *>(f);
    if (sf->pos >= sf->src_len)
      return 0;
    size_t avail = sf->src_len - sf->pos;
    size_t to_read = len < sf->max_read ? len : sf->max_read;
    if (to_read > avail)
      to_read = avail;
    for (size_t i = 0; i < to_read; ++i)
      reinterpret_cast<char *>(data)[i] = sf->src[sf->pos + i];
    sf->pos += to_read;
    return to_read;
  }

  static ErrorOr<off_t> short_seek(TestFile *f, off_t offset, int whence) {
    ShortReadFile *sf = static_cast<ShortReadFile *>(f);
    if (whence == SEEK_SET)
      sf->pos = static_cast<size_t>(offset);
    return sf->pos;
  }

  static int short_close(TestFile *f) {
    delete reinterpret_cast<ShortReadFile *>(f);
    return 0;
  }

public:
  explicit ShortReadFile(char *buffer, size_t buflen, int bufmode,
                         const char *data, size_t data_len,
                         size_t max_read_bytes)
      : TestCustomFile({&short_write, &short_read, &short_seek, &short_close},
                       reinterpret_cast<uint8_t *>(buffer), buflen, bufmode,
                       false, FileMode("r")),
        src(data), src_len(data_len), pos(0), max_read(max_read_bytes) {}
};

TEST(LlvmLibcFileTest, ShortReadsUntilLengthOrEOF) {
  constexpr char DATA[] = "0123456789abcdef";
  constexpr size_t DATA_LEN = sizeof(DATA) - 1;
  LIBC_NAMESPACE::AllocChecker ac;

  // Test buffered mode (_IOFBF) with both > bufsize and <= bufsize reads when
  // platform_read returns at most 3 bytes per call (e.g. reading from a pipe).
  char file_buf[8];
  ShortReadFile *f_fbf =
      new (ac) ShortReadFile(file_buf, sizeof(file_buf), _IOFBF, DATA, DATA_LEN,
                             /*max_read_bytes=*/3);
  ASSERT_NE(f_fbf, nullptr);

  char out[32] = {0};
  // Read 5 bytes (<= bufsize): requires two 3-byte platform_read calls,
  // leaving 1 byte buffered and eof == false.
  auto res1 = f_fbf->read(out, 5);
  EXPECT_EQ(res1.value, size_t(5));
  EXPECT_FALSE(f_fbf->iseof());
  EXPECT_FALSE(f_fbf->error());

  // Read remaining 11 bytes + try to read past end (> bufsize): loops until
  // platform_read returns 0 and sets eof == true.
  auto res2 = f_fbf->read(out + 5, 20);
  EXPECT_EQ(res2.value, size_t(11));
  EXPECT_TRUE(f_fbf->iseof());
  EXPECT_FALSE(f_fbf->error());
  EXPECT_STREQ(out, DATA);
  ASSERT_EQ(f_fbf->close(), 0);

  // Test unbuffered mode (_IONBF) with short reads.
  ShortReadFile *f_nbf = new (ac)
      ShortReadFile(nullptr, 0, _IONBF, DATA, DATA_LEN, /*max_read_bytes=*/3);
  ASSERT_NE(f_nbf, nullptr);

  char out_nbf[32] = {0};
  auto res3 = f_nbf->read(out_nbf, 10);
  EXPECT_EQ(res3.value, size_t(10));
  EXPECT_FALSE(f_nbf->iseof());
  auto res4 = f_nbf->read(out_nbf + 10, 10);
  EXPECT_EQ(res4.value, size_t(6));
  EXPECT_TRUE(f_nbf->iseof());
  EXPECT_STREQ(out_nbf, DATA);
  ASSERT_EQ(f_nbf->close(), 0);
}

TEST(LlvmLibcFileTest, RecursiveFileLock) {
  char file_buf[16];
  StringFile *f =
      new_string_file(file_buf, sizeof(file_buf), _IOFBF, false, "w+");
  ASSERT_NE(f, nullptr);

  // Simulate flockfile(f) followed by locked operations (write, ungetc, error,
  // iseof) on the same thread, plus nested lock()/unlock().
  f->lock();
  f->lock();
  ASSERT_EQ(f->write("ab", 2).value, size_t(2));
  ASSERT_EQ(f->flush(), 0);
  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);

  char c = 0;
  ASSERT_EQ(f->read_unlocked(&c, 1).value, size_t(1));
  EXPECT_EQ(c, 'a');
  EXPECT_EQ(f->ungetc('x'), int('x'));
  EXPECT_FALSE(f->error());
  EXPECT_FALSE(f->iseof());

  char out[3] = {0};
  ASSERT_EQ(f->read(out, 2).value, size_t(2));
  EXPECT_STREQ(out, "xb");
  f->unlock();
  f->unlock();

  ASSERT_EQ(f->close(), 0);
}

TEST(LlvmLibcFileTest, StaticDispatch) {
  constexpr char PAYLOAD[] = "static dispatch payload";
  char stream_buf[16];
  size_t writes = 0;
  size_t reads = 0;
  size_t seeks = 0;
  size_t closes = 0;

  LIBC_NAMESPACE::AllocChecker ac;
  auto *f = new (ac) StaticBufferFile(stream_buf, sizeof(stream_buf), _IOFBF,
                                      /*owned=*/false, FileMode("w+"), &writes,
                                      &reads, &seeks, &closes);
  ASSERT_NE(f, nullptr);
  scope_exit close_f([&] { EXPECT_EQ(f->close(), 0); });
  EXPECT_EQ(static_cast<uint8_t>(f->get_dispatch_kind()),
            static_cast<uint8_t>(TestFile::DispatchKind::STATIC));
  EXPECT_FALSE(f->is_dynamic());

  ASSERT_EQ(f->write("hi", 2).value, size_t(2));
  EXPECT_EQ(writes, size_t(0)); // Buffered before flush.
  ASSERT_EQ(f->flush(), 0);
  EXPECT_EQ(writes, size_t(1));

  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);
  ASSERT_EQ(f->write(PAYLOAD, sizeof(PAYLOAD)).value, sizeof(PAYLOAD));
  ASSERT_EQ(f->flush(), 0);
  EXPECT_STREQ(f->get_str(), PAYLOAD);

  ASSERT_EQ(f->seek(0, SEEK_SET).value(), 0);
  EXPECT_EQ(f->tell().value(), off_t(0));

  char out[sizeof(PAYLOAD)] = {0};
  ASSERT_EQ(f->read(out, sizeof(PAYLOAD)).value, sizeof(PAYLOAD));
  EXPECT_STREQ(out, PAYLOAD);
  EXPECT_GT(reads, size_t(0));
  EXPECT_GT(seeks, size_t(0));

  char extra = 0;
  EXPECT_EQ(f->read(&extra, 1).value, size_t(0));
  EXPECT_TRUE(f->iseof());
  ASSERT_EQ(f->ungetc('!'), static_cast<int>('!'));
  EXPECT_FALSE(f->iseof());
  ASSERT_EQ(f->read(&extra, 1).value, size_t(1));
  EXPECT_EQ(extra, '!');

  close_f.release();
  ASSERT_EQ(f->close(), 0);
  EXPECT_EQ(closes, size_t(1));
}

namespace {

struct TestCookieState {
  char buf[64] = {0};
  size_t len = 0;
  size_t pos = 0;
  size_t close_calls = 0;
  int close_ret = 0;
};

} // namespace

static ssize_t test_cookie_read(void *c, char *out, size_t size) {
  auto *st = static_cast<TestCookieState *>(c);
  if (st->pos >= st->len)
    return 0;
  size_t avail = st->len - st->pos;
  size_t n = size < avail ? size : avail;
  for (size_t i = 0; i < n; ++i)
    out[i] = st->buf[st->pos + i];
  st->pos += n;
  return static_cast<ssize_t>(n);
}

static ssize_t test_cookie_write(void *c, const char *in, size_t size) {
  auto *st = static_cast<TestCookieState *>(c);
  size_t i = 0;
  for (; i < size && st->pos < sizeof(st->buf); ++i, ++st->pos)
    st->buf[st->pos] = in[i];
  if (st->pos > st->len)
    st->len = st->pos;
  return static_cast<ssize_t>(i);
}

static int test_cookie_seek(void *c, off_t *offset, int whence) {
  auto *st = static_cast<TestCookieState *>(c);
  off_t base = 0;
  if (whence == SEEK_SET)
    base = 0;
  else if (whence == SEEK_CUR)
    base = static_cast<off_t>(st->pos);
  else if (whence == SEEK_END)
    base = static_cast<off_t>(st->len);
  else
    return -1;
  off_t next = base + *offset;
  if (next < 0 || static_cast<size_t>(next) > sizeof(st->buf))
    return -1;
  st->pos = static_cast<size_t>(next);
  *offset = next;
  return 0;
}

static int test_cookie_close(void *c) {
  auto *st = static_cast<TestCookieState *>(c);
  ++st->close_calls;
  return st->close_ret;
}

static ssize_t test_cookie_fail_read(void *, char *, size_t) { return -1; }
static ssize_t test_cookie_fail_write(void *, const char *, size_t) {
  return -1;
}
static ssize_t test_cookie_zero_write(void *, const char *, size_t) {
  return 0;
}
static int test_cookie_fail_seek(void *, off_t *, int) { return -1; }
static ssize_t test_cookie_oversize_read(void *, char *, size_t size) {
  return static_cast<ssize_t>(size + 1);
}
static ssize_t test_cookie_oversize_write(void *, const char *, size_t size) {
  return static_cast<ssize_t>(size + 1);
}

TEST(LlvmLibcFileTest, StaticCustomAndCookieCoexist) {
  char static_buf[32];
  size_t static_writes = 0;
  size_t static_reads = 0;
  size_t static_seeks = 0;
  size_t static_closes = 0;

  LIBC_NAMESPACE::AllocChecker ac;
  auto *static_f = new (ac) StaticBufferFile(
      static_buf, sizeof(static_buf), _IOFBF, /*owned=*/false, FileMode("w+"),
      &static_writes, &static_reads, &static_seeks, &static_closes);
  ASSERT_NE(static_f, nullptr);
  scope_exit close_static([&] { EXPECT_EQ(static_f->close(), 0); });
  EXPECT_EQ(static_cast<uint8_t>(static_f->get_dispatch_kind()),
            static_cast<uint8_t>(TestFile::DispatchKind::STATIC));
  EXPECT_FALSE(static_f->is_dynamic());

  constexpr char STATIC_MSG[] = "static fast path";
  ASSERT_EQ(static_f->write(STATIC_MSG, sizeof(STATIC_MSG)).value,
            sizeof(STATIC_MSG));
  ASSERT_EQ(static_f->flush(), 0);
  EXPECT_EQ(static_writes, size_t(1));
  ASSERT_EQ(static_f->seek(0, SEEK_SET).value(), 0);
  EXPECT_EQ(static_seeks, size_t(1));

  char read_back[sizeof(STATIC_MSG)] = {0};
  ASSERT_EQ(static_f->read(read_back, sizeof(STATIC_MSG)).value,
            sizeof(STATIC_MSG));
  EXPECT_STREQ(read_back, STATIC_MSG);
  EXPECT_EQ(static_reads, size_t(1));

  // Simultaneously exercise a CUSTOM stream and a COOKIE stream via TestFile *
  // and confirm neither invokes TestFilePlatform.
  char custom_buf[32];
  StringFile *custom_f =
      new_string_file(custom_buf, sizeof(custom_buf), _IOFBF, false, "w+");
  ASSERT_NE(custom_f, nullptr);
  scope_exit close_custom([&] { EXPECT_EQ(custom_f->close(), 0); });
  EXPECT_EQ(static_cast<uint8_t>(custom_f->get_dispatch_kind()),
            static_cast<uint8_t>(TestFile::DispatchKind::CUSTOM));
  EXPECT_TRUE(custom_f->is_dynamic());

  TestCookieState cookie_state;
  uint8_t cookie_buf[32];
  TestFile *cookie_f = new (ac) TestCookieFile(
      &cookie_state,
      CookieOps{&test_cookie_read, &test_cookie_write, &test_cookie_seek,
                &test_cookie_close},
      cookie_buf, sizeof(cookie_buf), _IOFBF, /*owned=*/false, FileMode("w+"));
  ASSERT_NE(cookie_f, nullptr);
  scope_exit close_cookie([&] { EXPECT_EQ(cookie_f->close(), 0); });
  EXPECT_EQ(static_cast<uint8_t>(cookie_f->get_dispatch_kind()),
            static_cast<uint8_t>(TestFile::DispatchKind::COOKIE));
  EXPECT_TRUE(cookie_f->is_dynamic());

  constexpr char CUSTOM_MSG[] = "custom ops path";
  ASSERT_EQ(custom_f->write(CUSTOM_MSG, sizeof(CUSTOM_MSG)).value,
            sizeof(CUSTOM_MSG));
  ASSERT_EQ(custom_f->flush(), 0);
  EXPECT_STREQ(custom_f->get_str(), CUSTOM_MSG);

  constexpr char COOKIE_MSG[] = "cookie ops path";
  ASSERT_EQ(cookie_f->write(COOKIE_MSG, sizeof(COOKIE_MSG)).value,
            sizeof(COOKIE_MSG));
  ASSERT_EQ(cookie_f->seek(0, SEEK_SET).value(), 0);
  char cookie_out[sizeof(COOKIE_MSG)] = {0};
  ASSERT_EQ(cookie_f->read(cookie_out, sizeof(COOKIE_MSG)).value,
            sizeof(COOKIE_MSG));
  EXPECT_STREQ(cookie_out, COOKIE_MSG);

  EXPECT_EQ(static_writes, size_t(1));
  EXPECT_EQ(static_reads, size_t(1));
  EXPECT_EQ(static_seeks, size_t(1));

  close_custom.release();
  ASSERT_EQ(custom_f->close(), 0);
  close_cookie.release();
  ASSERT_EQ(cookie_f->close(), 0);
  EXPECT_EQ(cookie_state.close_calls, size_t(1));
  EXPECT_EQ(static_closes, size_t(0));

  close_static.release();
  ASSERT_EQ(static_f->close(), 0);
  EXPECT_EQ(static_closes, size_t(1));
}

TEST(LlvmLibcFileTest, CustomNullHookFallback) {
  uint8_t stream_buf[16];
  TestCustomFile f({nullptr, nullptr, nullptr, nullptr}, stream_buf,
                   sizeof(stream_buf), _IOFBF, /*owned=*/false, FileMode("r+"));
  EXPECT_EQ(static_cast<uint8_t>(f.get_dispatch_kind()),
            static_cast<uint8_t>(TestFile::DispatchKind::CUSTOM));
  EXPECT_TRUE(f.is_dynamic());

  // default_write accepts and discards bytes without setting the error flag.
  constexpr char DATA[] = "discarded output exceeding buffer";
  auto write_res = f.write(DATA, sizeof(DATA));
  EXPECT_FALSE(write_res.has_error());
  EXPECT_EQ(write_res.value, sizeof(DATA));
  EXPECT_EQ(f.flush(), 0);
  EXPECT_FALSE(f.error());

  // default_seek reports ESPIPE for non-seekable streams.
  auto seek_res = f.seek(0, SEEK_SET);
  ASSERT_FALSE(seek_res.has_value());
  EXPECT_EQ(seek_res.error(), ESPIPE);

  auto tell_res = f.tell();
  ASSERT_FALSE(tell_res.has_value());
  EXPECT_EQ(tell_res.error(), ESPIPE);

  // Test default_read returning EOF (0 bytes, no error).
  TestCustomFile rf({nullptr, nullptr, nullptr, nullptr}, stream_buf,
                    sizeof(stream_buf), _IOFBF, /*owned=*/false,
                    FileMode("r+"));
  char out[8] = {0};
  auto read_res = rf.read(out, sizeof(out));
  EXPECT_FALSE(read_res.has_error());
  EXPECT_EQ(read_res.value, size_t(0));
  EXPECT_TRUE(rf.iseof());
  EXPECT_FALSE(rf.error());

  // ungetc still works on top of a stream with null read hook.
  ASSERT_EQ(rf.ungetc('Q'), static_cast<int>('Q'));
  EXPECT_FALSE(rf.iseof());
  ASSERT_EQ(rf.read(out, 1).value, size_t(1));
  EXPECT_EQ(out[0], 'Q');

  // default_close is a no-op returning 0.
  EXPECT_EQ(f.close(), 0);
  EXPECT_EQ(rf.close(), 0);
}

TEST(LlvmLibcFileTest, CookieNullHooksAndErrorMapping) {
  LIBC_NAMESPACE::AllocChecker ac;
  uint8_t stream_buf[16];

  // Null hooks: read -> 0 (EOF), seek -> ESPIPE, write -> discards output,
  // close -> remove_file + 0 + delete this.
  TestFile *null_f = new (ac) TestCookieFile(
      nullptr, CookieOps{nullptr, nullptr, nullptr, nullptr}, stream_buf,
      sizeof(stream_buf), _IOFBF, /*owned=*/false, FileMode("r+"));
  ASSERT_NE(null_f, nullptr);
  TestFile::add_file(null_f);
  scope_exit close_null([&] { EXPECT_EQ(null_f->close(), 0); });

  char out[8] = {0};
  auto read_res = null_f->read(out, sizeof(out));
  EXPECT_FALSE(read_res.has_error());
  EXPECT_EQ(read_res.value, size_t(0));
  EXPECT_TRUE(null_f->iseof());
  EXPECT_FALSE(null_f->error());

  auto seek_res = null_f->seek(0, SEEK_SET);
  ASSERT_FALSE(seek_res.has_value());
  EXPECT_EQ(seek_res.error(), ESPIPE);

  // Per fopencookie(3), a null write hook discards output without error.
  auto null_write_res = null_f->write("abc", 3);
  EXPECT_FALSE(null_write_res.has_error());
  EXPECT_EQ(null_write_res.value, size_t(3));
  EXPECT_EQ(null_f->flush(), 0);
  EXPECT_FALSE(null_f->error());

  // close() unlinks from list_all, returns 0, and deletes |null_f|.
  EXPECT_EQ(TestFile::get_first_file(), null_f);
  close_null.release();
  EXPECT_EQ(null_f->close(), 0);
  EXPECT_EQ(TestFile::get_first_file(), nullptr);

  // Buffered (_IOFBF) stream with a write hook returning 0 (error per
  // fopencookie(3)): buffered write defers until flush/seek, which must report
  // EIO instead of silently succeeding.
  TestFile *zero_write_f = new (ac) TestCookieFile(
      nullptr, CookieOps{nullptr, &test_cookie_zero_write, nullptr, nullptr},
      stream_buf, sizeof(stream_buf), _IOFBF, /*owned=*/false, FileMode("r+"));
  ASSERT_NE(zero_write_f, nullptr);
  scope_exit close_zero_write([&] {
    (void)zero_write_f->read(out, 1);
    EXPECT_EQ(zero_write_f->close(), 0);
  });

  ASSERT_EQ(zero_write_f->write("abc", 3).value, size_t(3));
  EXPECT_EQ(zero_write_f->flush(), EIO);
  EXPECT_TRUE(zero_write_f->error());
  zero_write_f->clearerr();

  auto buf_seek_res = zero_write_f->seek(0, SEEK_SET);
  ASSERT_FALSE(buf_seek_res.has_value());
  EXPECT_EQ(buf_seek_res.error(), EIO);
  EXPECT_TRUE(zero_write_f->error());

  // Switch out of WRITE mode before closing so close() deletes |zero_write_f|.
  (void)zero_write_f->read(out, 1);
  close_zero_write.release();
  EXPECT_EQ(zero_write_f->close(), 0);

  // Failing callbacks: read/write/seek/close map failures to EIO, and close()
  // still deletes |fail_f| when ops.close returns non-zero.
  TestCookieState fail_state;
  fail_state.close_ret = -42;
  TestFile *fail_f = new (ac) TestCookieFile(
      &fail_state,
      CookieOps{&test_cookie_fail_read, &test_cookie_fail_write,
                &test_cookie_fail_seek, &test_cookie_close},
      stream_buf, sizeof(stream_buf), _IOFBF, /*owned=*/false, FileMode("w+"));
  ASSERT_NE(fail_f, nullptr);
  scope_exit close_fail([&] { EXPECT_EQ(fail_f->close(), EIO); });

  auto fail_read = fail_f->read(out, sizeof(out));
  EXPECT_TRUE(fail_read.has_error());
  EXPECT_EQ(fail_read.error, EIO);
  EXPECT_TRUE(fail_f->error());
  fail_f->clearerr();

  constexpr char LARGE_WRITE[36] = "exceeds two 16-byte stream buffers!";
  auto fail_write = fail_f->write(LARGE_WRITE, sizeof(LARGE_WRITE));
  EXPECT_TRUE(fail_write.has_error());
  EXPECT_EQ(fail_write.error, EIO);
  EXPECT_TRUE(fail_f->error());
  fail_f->clearerr();

  auto fail_seek = fail_f->seek(0, SEEK_SET);
  ASSERT_FALSE(fail_seek.has_value());
  EXPECT_EQ(fail_seek.error(), EIO);

  close_fail.release();
  EXPECT_EQ(fail_f->close(), EIO);
  EXPECT_EQ(fail_state.close_calls, size_t(1));

  // Oversize callbacks (> requested size) are rejected as EIO rather than
  // corrupting stream buffer bounds.
  TestFile *oversize_f = new (ac) TestCookieFile(
      nullptr,
      CookieOps{&test_cookie_oversize_read, &test_cookie_oversize_write,
                nullptr, nullptr},
      stream_buf, sizeof(stream_buf), _IOFBF, /*owned=*/false, FileMode("w+"));
  ASSERT_NE(oversize_f, nullptr);
  scope_exit close_oversize([&] { EXPECT_EQ(oversize_f->close(), 0); });

  auto oversize_read = oversize_f->read(out, sizeof(out));
  EXPECT_TRUE(oversize_read.has_error());
  EXPECT_EQ(oversize_read.error, EIO);
  EXPECT_EQ(oversize_read.value, size_t(0));
  EXPECT_TRUE(oversize_f->error());
  oversize_f->clearerr();

  auto oversize_write = oversize_f->write(LARGE_WRITE, sizeof(LARGE_WRITE));
  EXPECT_TRUE(oversize_write.has_error());
  EXPECT_EQ(oversize_write.error, EIO);
  EXPECT_TRUE(oversize_f->error());
  oversize_f->clearerr();

  close_oversize.release();
  EXPECT_EQ(oversize_f->close(), 0);
}
