//===-- Unittests for file operations like fopen, flcose etc --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/clearerr.h"
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fputs.h"
#include "src/stdio/fread.h"
#include "src/stdio/fseek.h"
#include "src/stdio/fwrite.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::EQ;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::NE;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::returns;

TEST(LlvmLibcFILETest, SimpleFileOperations) {
  constexpr char FILENAME[] = "testdata/simple_operations.test";
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  constexpr char CONTENT[] = "1234567890987654321";
  ASSERT_EQ(sizeof(CONTENT) - 1,
            LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file));

  // This is not a readable file.
  char read_data[sizeof(CONTENT)];
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_data, 1, sizeof(CONTENT), file),
              returns(EQ(size_t(0))).with_errno(NE(0)));
  ASSERT_NE(LIBC_NAMESPACE::ferror(file), 0);
  libc_errno = 0;

  LIBC_NAMESPACE::clearerr(file);
  ASSERT_EQ(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  constexpr size_t READ_SIZE = 5;
  char data[READ_SIZE];
  data[READ_SIZE - 1] = '\0';
  ASSERT_EQ(LIBC_NAMESPACE::fread(data, 1, READ_SIZE - 1, file), READ_SIZE - 1);
  ASSERT_STREQ(data, "1234");
  ASSERT_EQ(LIBC_NAMESPACE::fseek(file, 5, SEEK_CUR), 0);
  ASSERT_EQ(LIBC_NAMESPACE::fread(data, 1, READ_SIZE - 1, file), READ_SIZE - 1);
  ASSERT_STREQ(data, "0987");
  ASSERT_EQ(LIBC_NAMESPACE::fseek(file, -5, SEEK_CUR), 0);
  ASSERT_EQ(LIBC_NAMESPACE::fread(data, 1, READ_SIZE - 1, file), READ_SIZE - 1);
  ASSERT_STREQ(data, "9098");

  // Reading another time should trigger eof.
  ASSERT_NE(sizeof(CONTENT),
            LIBC_NAMESPACE::fread(read_data, 1, sizeof(CONTENT), file));
  ASSERT_NE(LIBC_NAMESPACE::feof(file), 0);

  // Should be an error to write.
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT), file),
              returns(EQ(size_t(0))).with_errno(NE(0)));
  ASSERT_NE(LIBC_NAMESPACE::ferror(file), 0);
  libc_errno = 0;

  LIBC_NAMESPACE::clearerr(file);

  // Should be an error to puts.
  ASSERT_THAT(LIBC_NAMESPACE::fputs(CONTENT, file),
              returns(EQ(EOF)).with_errno(NE(0)));
  ASSERT_NE(LIBC_NAMESPACE::ferror(file), 0);
  libc_errno = 0;

  LIBC_NAMESPACE::clearerr(file);
  ASSERT_EQ(LIBC_NAMESPACE::ferror(file), 0);

  libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::fwrite("nothing", 1, 1, file),
              returns(EQ(0)).with_errno(NE(0)));
  libc_errno = 0;

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Now try puts.
  file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  // fputs returns a negative value on error (EOF) or any non-negative value on
  // success. This assert checks that the return value is non-negative.
  ASSERT_GE(LIBC_NAMESPACE::fputs(CONTENT, file), 0);

  LIBC_NAMESPACE::clearerr(file);
  ASSERT_EQ(LIBC_NAMESPACE::ferror(file), 0);

  // This is not a readable file.
  libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::fread(data, 1, 1, file),
              returns(EQ(0)).with_errno(NE(0)));
  libc_errno = 0;

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::fread(read_data, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  read_data[sizeof(CONTENT) - 1] = '\0';
  ASSERT_STREQ(read_data, CONTENT);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Check that the other functions correctly set libc_errno.

  // libc_errno = 0;
  // ASSERT_NE(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), 0);
  // ASSERT_ERRNO_FAILURE();

  // libc_errno = 0;
  // ASSERT_NE(LIBC_NAMESPACE::fclose(file), 0);
  // ASSERT_ERRNO_FAILURE();

  // libc_errno = 0;
  // ASSERT_EQ(LIBC_NAMESPACE::fopen("INVALID FILE NAME", "r"),
  //           static_cast<FILE *>(nullptr));
  // ASSERT_ERRNO_FAILURE();
}

TEST(LlvmLibcFILETest, FFlush) {
  constexpr char FILENAME[] = "testdata/fflush.test";
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);
  constexpr char CONTENT[] = "1234567890987654321";
  ASSERT_EQ(sizeof(CONTENT),
            LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT), file));

  // Flushing at this point should write the data to disk. So, we should be
  // able to read it back.
  ASSERT_EQ(0, LIBC_NAMESPACE::fflush(file));

  char data[sizeof(CONTENT)];
  ASSERT_EQ(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), 0);
  ASSERT_EQ(LIBC_NAMESPACE::fread(data, 1, sizeof(CONTENT), file),
            sizeof(CONTENT));
  ASSERT_STREQ(data, CONTENT);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcFILETest, FOpenFWriteSizeGreaterThanOne) {
  using MyStruct = struct {
    char c;
    unsigned long long i;
  };
  constexpr MyStruct WRITE_DATA[] = {{'a', 1}, {'b', 2}, {'c', 3}};
  constexpr size_t WRITE_NMEMB = sizeof(WRITE_DATA) / sizeof(MyStruct);
  constexpr char FILENAME[] = "testdata/fread_fwrite.test";

  libc_errno = 0;
  FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(size_t(0), LIBC_NAMESPACE::fwrite(WRITE_DATA, 0, 1, file));
  ASSERT_THAT(
      LIBC_NAMESPACE::fwrite(WRITE_DATA, sizeof(MyStruct), WRITE_NMEMB, file),
      returns(EQ(WRITE_NMEMB)).with_errno(EQ(0)));
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  MyStruct read_data[WRITE_NMEMB];
  ASSERT_EQ(size_t(0), LIBC_NAMESPACE::fread(read_data, 0, 1, file));
  ASSERT_THAT(
      LIBC_NAMESPACE::fread(read_data, sizeof(MyStruct), WRITE_NMEMB, file),
      returns(EQ(WRITE_NMEMB)).with_errno(EQ(0)));
  // Trying to read more should fetch nothing.
  ASSERT_THAT(
      LIBC_NAMESPACE::fread(read_data, sizeof(MyStruct), WRITE_NMEMB, file),
      returns(EQ(0)).with_errno(EQ(0)));
  EXPECT_NE(LIBC_NAMESPACE::feof(file), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(file), 0);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
  // Verify that the data which was read is correct.
  for (size_t i = 0; i < WRITE_NMEMB; ++i) {
    ASSERT_EQ(read_data[i].c, WRITE_DATA[i].c);
    ASSERT_EQ(read_data[i].i, WRITE_DATA[i].i);
  }
}
