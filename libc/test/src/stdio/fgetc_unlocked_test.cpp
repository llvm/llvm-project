//===-- Unittests for fgetc -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/feof_unlocked.h"
#include "src/stdio/ferror.h"
#include "src/stdio/ferror_unlocked.h"
#include "src/stdio/fgetc_unlocked.h"
#include "src/stdio/flockfile.h"
#include "src/stdio/fopen.h"
#include "src/stdio/funlockfile.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/getc_unlocked.h"
#include "test/UnitTest/Test.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

class LlvmLibcGetcTest : public __llvm_libc::testing::Test {
public:
  using GetcFunc = int(FILE *);
  void test_with_func(GetcFunc *func, const char *filename) {
    ::FILE *file = __llvm_libc::fopen(filename, "w");
    ASSERT_FALSE(file == nullptr);
    constexpr char CONTENT[] = "123456789";
    constexpr size_t WRITE_SIZE = sizeof(CONTENT) - 1;
    ASSERT_EQ(WRITE_SIZE, __llvm_libc::fwrite(CONTENT, 1, WRITE_SIZE, file));
    // This is a write-only file so reads should fail.
    ASSERT_EQ(func(file), EOF);
    // This is an error and not a real EOF.
    ASSERT_EQ(__llvm_libc::feof(file), 0);
    ASSERT_NE(__llvm_libc::ferror(file), 0);
    libc_errno = 0;

    ASSERT_EQ(0, __llvm_libc::fclose(file));

    file = __llvm_libc::fopen(filename, "r");
    ASSERT_FALSE(file == nullptr);

    __llvm_libc::flockfile(file);
    for (size_t i = 0; i < WRITE_SIZE; ++i) {
      int c = func(file);
      ASSERT_EQ(c, int('1' + i));
    }
    // Reading more should return EOF but not set error.
    ASSERT_EQ(func(file), EOF);
    ASSERT_NE(__llvm_libc::feof_unlocked(file), 0);
    ASSERT_EQ(__llvm_libc::ferror_unlocked(file), 0);

    __llvm_libc::funlockfile(file);
    ASSERT_EQ(0, __llvm_libc::fclose(file));
  }
};

TEST_F(LlvmLibcGetcTest, WriteAndReadCharactersWithFgetcUnlocked) {
  test_with_func(&__llvm_libc::fgetc_unlocked, "testdata/fgetc_unlocked.test");
}

TEST_F(LlvmLibcGetcTest, WriteAndReadCharactersWithGetcUnlocked) {
  test_with_func(&__llvm_libc::getc_unlocked, "testdata/getc_unlocked.test");
}
