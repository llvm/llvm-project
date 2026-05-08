//===-- Unittests for file operations like fopen, flcose etc --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/rewind.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcRewindTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcRewindTest, WriteRewindRead) {
  constexpr char FILENAME[] = APPEND_LIBC_TEST("testdata/rewind.test");
  auto FILEPATH = libc_make_test_file_path(FILENAME);
  ::FILE *file = LIBC_NAMESPACE::fopen(FILEPATH, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr char FIRST_DATA[] = "123456789";
  ASSERT_THAT(LIBC_NAMESPACE::fwrite(FIRST_DATA, 1, sizeof(FIRST_DATA), file),
              Succeeds(sizeof(FIRST_DATA)));

  // File state is "123456789"

  LIBC_NAMESPACE::rewind(file);

  // Cursor is back to the start

  constexpr char SECOND_DATA[] = "abc";
  ASSERT_THAT(
      LIBC_NAMESPACE::fwrite(SECOND_DATA, 1, sizeof(SECOND_DATA) - 1, file),
      Succeeds(sizeof(SECOND_DATA) - 1));

  // File state is "abc456789"

  // attempt to read from write-only file causing error state
  char read_data[sizeof(FIRST_DATA)];
  ASSERT_EQ(LIBC_NAMESPACE::fread(read_data, 1, sizeof(read_data), file),
            size_t(0));
  ASSERT_ERRNO_FAILURE();
  ASSERT_NE(LIBC_NAMESPACE::ferror(file), 0);

  // rewind to start and check that that clears the error.
  LIBC_NAMESPACE::rewind(file);
  ASSERT_EQ(LIBC_NAMESPACE::ferror(file), 0);

  // Close out the file and reopen in read mode.
  LIBC_NAMESPACE::fclose(file);
  file = LIBC_NAMESPACE::fopen(FILEPATH, "r");
  ASSERT_FALSE(file == nullptr);

  // Read the file to check that it was written correctly.
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_data, 1, 3, file),
              Succeeds(size_t(3)));
  read_data[3] = '\0';
  ASSERT_STREQ(read_data, "abc");
  ASSERT_EQ(LIBC_NAMESPACE::ferror(file), 0);

  // check that rewind also works on read files.
  LIBC_NAMESPACE::rewind(file);
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_data, 1, sizeof(read_data), file),
              Succeeds(sizeof(FIRST_DATA)));
  read_data[sizeof(FIRST_DATA) - 1] = '\0';
  ASSERT_STREQ(read_data, "abc456789");

  LIBC_NAMESPACE::fclose(file);
}
