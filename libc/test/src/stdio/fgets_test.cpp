//===-- Unittests for fgets -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fgets.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"
#include "test/UnitTest/Test.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

TEST(LlvmLibcFgetsTest, WriteAndReadCharacters) {
  constexpr char FILENAME[] = "testdata/fgets.test";
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  constexpr char CONTENT[] = "123456789\n"
                             "1234567\n"
                             "123456\n"
                             "1";
  constexpr size_t WRITE_SIZE = sizeof(CONTENT) - 1;

  char buff[8];
  char *output;

  ASSERT_EQ(WRITE_SIZE, LIBC_NAMESPACE::fwrite(CONTENT, 1, WRITE_SIZE, file));
  // This is a write-only file so reads should fail.
  ASSERT_TRUE(LIBC_NAMESPACE::fgets(buff, 8, file) == nullptr);
  // This is an error and not a real EOF.
  ASSERT_EQ(LIBC_NAMESPACE::feof(file), 0);
  ASSERT_NE(LIBC_NAMESPACE::ferror(file), 0);
  libc_errno = 0;

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // If we request just 1 byte, it should return just a null byte and not
  // advance the read head. This is implementation defined.
  output = LIBC_NAMESPACE::fgets(buff, 1, file);
  ASSERT_TRUE(output == buff);
  ASSERT_EQ(buff[0], '\0');
  ASSERT_ERRNO_SUCCESS();

  // If we request less than 1 byte, it should do nothing and return nullptr.
  // This is also implementation defined.
  output = LIBC_NAMESPACE::fgets(buff, 0, file);
  ASSERT_TRUE(output == nullptr);

  const char *output_arr[] = {
      "1234567", "89\n", "1234567", "\n", "123456\n", "1",
  };

  constexpr size_t ARR_SIZE = sizeof(output_arr) / sizeof(char *);

  for (size_t i = 0; i < ARR_SIZE; ++i) {
    output = LIBC_NAMESPACE::fgets(buff, 8, file);

    // This pointer comparison is intentional, fgets should return a pointer to
    // buff when it succeeds.
    ASSERT_TRUE(output == buff);
    ASSERT_EQ(LIBC_NAMESPACE::ferror(file), 0);

    EXPECT_STREQ(buff, output_arr[i]);
  }

  // This should have hit the end of the file, but that isn't an error unless it
  // fails to read anything.
  ASSERT_NE(LIBC_NAMESPACE::feof(file), 0);
  ASSERT_EQ(LIBC_NAMESPACE::ferror(file), 0);
  ASSERT_ERRNO_SUCCESS();

  // Reading more should be an EOF, but not an error.
  output = LIBC_NAMESPACE::fgets(buff, 8, file);
  ASSERT_TRUE(output == nullptr);
  ASSERT_NE(LIBC_NAMESPACE::feof(file), 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));
}
