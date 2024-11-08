//===-- Unittests for fchmod ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/fchmod.h"
#include "src/unistd/close.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/fcntl_macros.h"
#include <sys/stat.h>

TEST(LlvmLibcChmodTest, ChangeAndOpen) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  // The test file is initially writable. We open it for writing and ensure
  // that it indeed can be opened for writing. Next, we close the file and
  // make it readonly using chmod. We test that chmod actually succeeded by
  // trying to open the file for writing and failing.
  constexpr const char *TEST_FILE = "testdata/fchmod.test";
  const char WRITE_DATA[] = "test data";
  constexpr ssize_t WRITE_SIZE = ssize_t(sizeof(WRITE_DATA));
  LIBC_NAMESPACE::libc_errno = 0;

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_APPEND | O_WRONLY);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::write(fd, WRITE_DATA, sizeof(WRITE_DATA)),
            WRITE_SIZE);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  fd = LIBC_NAMESPACE::open(TEST_FILE, O_APPEND | O_WRONLY);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_THAT(LIBC_NAMESPACE::fchmod(fd, S_IRUSR), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  // Opening for writing should fail.
  EXPECT_EQ(LIBC_NAMESPACE::open(TEST_FILE, O_APPEND | O_WRONLY), -1);
  ASSERT_ERRNO_FAILURE();
  LIBC_NAMESPACE::libc_errno = 0;
  // But opening for reading should succeed.
  fd = LIBC_NAMESPACE::open(TEST_FILE, O_APPEND | O_RDONLY);
  EXPECT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();

  EXPECT_THAT(LIBC_NAMESPACE::fchmod(fd, S_IRWXU), Succeeds(0));
  EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST(LlvmLibcChmodTest, NonExistentFile) {
  LIBC_NAMESPACE::libc_errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::fchmod(-1, S_IRUSR), -1);
  ASSERT_ERRNO_FAILURE();
  LIBC_NAMESPACE::libc_errno = 0;
}
