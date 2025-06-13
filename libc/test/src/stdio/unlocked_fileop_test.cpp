//===-- Unittests for f operations like fopen, flcose etc --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/clearerr_unlocked.h"
#include "src/stdio/fclose.h"
#include "src/stdio/feof_unlocked.h"
#include "src/stdio/ferror_unlocked.h"
#include "src/stdio/flockfile.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread_unlocked.h"
#include "src/stdio/funlockfile.h"
#include "src/stdio/fwrite_unlocked.h"
#include "test/UnitTest/Test.h"

#include "src/__support/libc_errno.h"

TEST(LlvmLibcFILETest, UnlockedReadAndWrite) {
  constexpr char fNAME[] = "testdata/unlocked_read_and_write.test";
  ::FILE *f = LIBC_NAMESPACE::fopen(fNAME, "w");
  ASSERT_FALSE(f == nullptr);
  constexpr char CONTENT[] = "1234567890987654321";
  LIBC_NAMESPACE::flockfile(f);
  ASSERT_EQ(sizeof(CONTENT) - 1, LIBC_NAMESPACE::fwrite_unlocked(
                                     CONTENT, 1, sizeof(CONTENT) - 1, f));
  // Should be an error to read.
  constexpr size_t READ_SIZE = 5;
  char data[READ_SIZE * 2 + 1];
  data[READ_SIZE * 2] = '\0';

  ASSERT_EQ(size_t(0),
            LIBC_NAMESPACE::fread_unlocked(data, 1, sizeof(READ_SIZE), f));
  ASSERT_NE(LIBC_NAMESPACE::ferror_unlocked(f), 0);
  ASSERT_ERRNO_FAILURE();
  libc_errno = 0;

  LIBC_NAMESPACE::clearerr_unlocked(f);
  ASSERT_EQ(LIBC_NAMESPACE::ferror_unlocked(f), 0);

  LIBC_NAMESPACE::funlockfile(f);
  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(f));

  f = LIBC_NAMESPACE::fopen(fNAME, "r");
  ASSERT_FALSE(f == nullptr);

  LIBC_NAMESPACE::flockfile(f);
  ASSERT_EQ(LIBC_NAMESPACE::fread_unlocked(data, 1, READ_SIZE, f), READ_SIZE);
  ASSERT_EQ(LIBC_NAMESPACE::fread_unlocked(data + READ_SIZE, 1, READ_SIZE, f),
            READ_SIZE);

  // Should be an error to write.
  ASSERT_EQ(size_t(0),
            LIBC_NAMESPACE::fwrite_unlocked(CONTENT, 1, sizeof(CONTENT), f));
  ASSERT_NE(LIBC_NAMESPACE::ferror_unlocked(f), 0);
  ASSERT_ERRNO_FAILURE();
  libc_errno = 0;

  LIBC_NAMESPACE::clearerr_unlocked(f);
  ASSERT_EQ(LIBC_NAMESPACE::ferror_unlocked(f), 0);

  // Reading more should trigger eof.
  char large_data[sizeof(CONTENT)];
  ASSERT_NE(sizeof(CONTENT),
            LIBC_NAMESPACE::fread_unlocked(large_data, 1, sizeof(CONTENT), f));
  ASSERT_NE(LIBC_NAMESPACE::feof_unlocked(f), 0);

  LIBC_NAMESPACE::funlockfile(f);
  ASSERT_STREQ(data, "1234567890");

  ASSERT_EQ(LIBC_NAMESPACE::fclose(f), 0);
}
