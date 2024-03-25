//===-- Unittests for fgetc -----------------------------------------------===//
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
#include "src/stdio/fgetc.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/getc.h"
#include "test/UnitTest/Test.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

class LlvmLibcGetcTest : public LIBC_NAMESPACE::testing::Test {
public:
  using GetcFunc = int(FILE *);
  void test_with_func(GetcFunc *func, const char *filename) {
    ::FILE *file = LIBC_NAMESPACE::fopen(filename, "w");
    ASSERT_FALSE(file == nullptr);
    constexpr char CONTENT[] = "123456789";
    constexpr size_t WRITE_SIZE = sizeof(CONTENT) - 1;
    ASSERT_EQ(WRITE_SIZE, LIBC_NAMESPACE::fwrite(CONTENT, 1, WRITE_SIZE, file));
    // This is a write-only file so reads should fail.
    ASSERT_EQ(func(file), EOF);
    // This is an error and not a real EOF.
    ASSERT_EQ(LIBC_NAMESPACE::feof(file), 0);
    ASSERT_NE(LIBC_NAMESPACE::ferror(file), 0);
    LIBC_NAMESPACE::libc_errno = 0;

    ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));

    file = LIBC_NAMESPACE::fopen(filename, "r");
    ASSERT_FALSE(file == nullptr);

    for (size_t i = 0; i < WRITE_SIZE; ++i) {
      int c = func(file);
      ASSERT_EQ(c, int('1' + i));
    }
    // Reading more should return EOF but not set error.
    ASSERT_EQ(func(file), EOF);
    ASSERT_NE(LIBC_NAMESPACE::feof(file), 0);
    ASSERT_EQ(LIBC_NAMESPACE::ferror(file), 0);

    ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file));
  }
};

TEST_F(LlvmLibcGetcTest, WriteAndReadCharactersWithFgetc) {
  test_with_func(&LIBC_NAMESPACE::fgetc, "testdata/fgetc.test");
}

TEST_F(LlvmLibcGetcTest, WriteAndReadCharactersWithGetc) {
  test_with_func(&LIBC_NAMESPACE::getc, "testdata/getc.test");
}
