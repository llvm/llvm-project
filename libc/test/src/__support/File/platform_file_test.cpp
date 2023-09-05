//===-- Unittests for target platform file implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"
#include "test/UnitTest/Test.h"

#include <stdio.h> // For SEEK_* macros

using File = __llvm_libc::File;
constexpr char TEXT[] = "Hello, File";
constexpr size_t TEXT_SIZE = sizeof(TEXT) - 1; // Ignore the null terminator

LIBC_INLINE File *openfile(const char *file_name, const char *mode) {
  auto error_or_file = __llvm_libc::openfile(file_name, mode);
  return error_or_file.has_value() ? error_or_file.value() : nullptr;
}

TEST(LlvmLibcPlatformFileTest, CreateWriteCloseAndReadBack) {
  constexpr char FILENAME[] = "testdata/create_write_close_and_readback.test";
  File *file = openfile(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(file->write(TEXT, TEXT_SIZE).value, TEXT_SIZE);
  ASSERT_EQ(file->close(), 0);

  file = openfile(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  char data[sizeof(TEXT)];
  ASSERT_EQ(file->read(data, TEXT_SIZE).value, TEXT_SIZE);
  data[TEXT_SIZE] = '\0';
  ASSERT_STREQ(data, TEXT);

  // Reading more data should trigger EOF.
  ASSERT_EQ(file->read(data, TEXT_SIZE).value, size_t(0));
  ASSERT_TRUE(file->iseof());

  ASSERT_EQ(file->close(), 0);
}

TEST(LlvmLibcPlatformFileTest, CreateWriteSeekAndReadBack) {
  constexpr char FILENAME[] = "testdata/create_write_seek_and_readback.test";
  File *file = openfile(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(file->write(TEXT, TEXT_SIZE).value, TEXT_SIZE);

  ASSERT_EQ(file->seek(0, SEEK_SET).value(), 0);

  char data[sizeof(TEXT)];
  ASSERT_EQ(file->read(data, TEXT_SIZE).value, TEXT_SIZE);
  data[TEXT_SIZE] = '\0';
  ASSERT_STREQ(data, TEXT);

  // Reading more data should trigger EOF.
  ASSERT_EQ(file->read(data, TEXT_SIZE).value, size_t(0));
  ASSERT_TRUE(file->iseof());

  ASSERT_EQ(file->close(), 0);
}

TEST(LlvmLibcPlatformFileTest, CreateAppendCloseAndReadBack) {
  constexpr char FILENAME[] = "testdata/create_append_close_and_readback.test";
  File *file = openfile(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(file->write(TEXT, TEXT_SIZE).value, TEXT_SIZE);
  ASSERT_EQ(file->close(), 0);

  file = openfile(FILENAME, "a");
  ASSERT_FALSE(file == nullptr);
  constexpr char APPEND_TEXT[] = " Append Text";
  constexpr size_t APPEND_TEXT_SIZE = sizeof(APPEND_TEXT) - 1;
  ASSERT_EQ(file->write(APPEND_TEXT, APPEND_TEXT_SIZE).value, APPEND_TEXT_SIZE);
  ASSERT_EQ(file->close(), 0);

  file = openfile(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  constexpr size_t READ_SIZE = TEXT_SIZE + APPEND_TEXT_SIZE;
  char data[READ_SIZE + 1];
  ASSERT_EQ(file->read(data, READ_SIZE).value, READ_SIZE);
  data[READ_SIZE] = '\0';
  ASSERT_STREQ(data, "Hello, File Append Text");

  // Reading more data should trigger EOF.
  ASSERT_EQ(file->read(data, READ_SIZE).value, size_t(0));
  ASSERT_TRUE(file->iseof());

  ASSERT_EQ(file->close(), 0);
}

TEST(LlvmLibcPlatformFileTest, CreateAppendSeekAndReadBack) {
  constexpr char FILENAME[] = "testdata/create_append_seek_and_readback.test";
  File *file = openfile(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(file->write(TEXT, TEXT_SIZE).value, TEXT_SIZE);
  ASSERT_EQ(file->close(), 0);

  file = openfile(FILENAME, "a+");
  ASSERT_FALSE(file == nullptr);
  constexpr char APPEND_TEXT[] = " Append Text";
  constexpr size_t APPEND_TEXT_SIZE = sizeof(APPEND_TEXT) - 1;
  ASSERT_EQ(file->write(APPEND_TEXT, APPEND_TEXT_SIZE).value, APPEND_TEXT_SIZE);

  ASSERT_EQ(file->seek(-APPEND_TEXT_SIZE, SEEK_END).value(), 0);
  char data[APPEND_TEXT_SIZE + 1];
  ASSERT_EQ(file->read(data, APPEND_TEXT_SIZE).value, APPEND_TEXT_SIZE);
  data[APPEND_TEXT_SIZE] = '\0';
  ASSERT_STREQ(data, APPEND_TEXT);

  // Reading more data should trigger EOF.
  ASSERT_EQ(file->read(data, APPEND_TEXT_SIZE).value, size_t(0));
  ASSERT_TRUE(file->iseof());

  ASSERT_EQ(file->close(), 0);
}

TEST(LlvmLibcPlatformFileTest, LargeFile) {
  constexpr size_t DATA_SIZE = File::DEFAULT_BUFFER_SIZE >> 2;
  constexpr char BYTE = 123;
  char write_data[DATA_SIZE];
  for (size_t i = 0; i < DATA_SIZE; ++i)
    write_data[i] = BYTE;

  constexpr char FILENAME[] = "testdata/large_file.test";
  File *file = openfile(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr int REPEAT = 5;
  for (int i = 0; i < REPEAT; ++i) {
    ASSERT_EQ(file->write(write_data, DATA_SIZE).value, DATA_SIZE);
  }
  ASSERT_EQ(file->close(), 0);

  file = openfile(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  constexpr size_t READ_SIZE = DATA_SIZE * REPEAT;
  char data[READ_SIZE] = {0};
  ASSERT_EQ(file->read(data, READ_SIZE).value, READ_SIZE);

  for (size_t i = 0; i < READ_SIZE; ++i)
    ASSERT_EQ(data[i], BYTE);

  // Reading more data should trigger EOF.
  ASSERT_EQ(file->read(data, 1).value, size_t(0));
  ASSERT_TRUE(file->iseof());

  ASSERT_EQ(file->close(), 0);
}

TEST(LlvmLibcPlatformFileTest, ReadSeekCurAndRead) {
  constexpr char FILENAME[] = "testdata/read_seek_cur_and_read.test";
  File *file = openfile(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  constexpr char CONTENT[] = "1234567890987654321";
  ASSERT_EQ(sizeof(CONTENT) - 1,
            file->write(CONTENT, sizeof(CONTENT) - 1).value);
  ASSERT_EQ(0, file->close());

  file = openfile(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  constexpr size_t READ_SIZE = 5;
  char data[READ_SIZE];
  data[READ_SIZE - 1] = '\0';
  ASSERT_EQ(file->read(data, READ_SIZE - 1).value, READ_SIZE - 1);
  ASSERT_STREQ(data, "1234");
  ASSERT_EQ(file->seek(5, SEEK_CUR).value(), 0);
  ASSERT_EQ(file->read(data, READ_SIZE - 1).value, READ_SIZE - 1);
  ASSERT_STREQ(data, "0987");
  ASSERT_EQ(file->seek(-5, SEEK_CUR).value(), 0);
  ASSERT_EQ(file->read(data, READ_SIZE - 1).value, READ_SIZE - 1);
  ASSERT_STREQ(data, "9098");

  ASSERT_EQ(file->close(), 0);
}

TEST(LlvmLibcPlatformFileTest, IncorrectOperation) {
  constexpr char FILENAME[] = "testdata/incorrect_operation.test";
  char data[1] = {123};

  File *file = openfile(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(file->read(data, 1).value, size_t(0)); // Cannot read
  ASSERT_FALSE(file->iseof());
  ASSERT_TRUE(file->error());
  ASSERT_EQ(file->close(), 0);

  file = openfile(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(file->write(data, 1).value, size_t(0)); // Cannot write
  ASSERT_FALSE(file->iseof());
  ASSERT_TRUE(file->error());
  ASSERT_EQ(file->close(), 0);

  file = openfile(FILENAME, "a");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(file->read(data, 1).value, size_t(0)); // Cannot read
  ASSERT_FALSE(file->iseof());
  ASSERT_TRUE(file->error());
  ASSERT_EQ(file->close(), 0);
}
