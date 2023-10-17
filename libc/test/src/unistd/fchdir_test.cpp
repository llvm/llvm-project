//===-- Unittests for fchdir ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/fchdir.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <fcntl.h>

TEST(LlvmLibcChdirTest, ChangeAndOpen) {
  // The idea of this test is that we will first open an existing test file
  // without changing the directory to make sure it exists. Next, we change
  // directory and open the same file to make sure that the "fchdir" operation
  // succeeded.
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE = "testdata/fchdir.test";
  constexpr const char *TEST_FILE_BASE = "fchdir.test";
  libc_errno = 0;

  int dir_fd = LIBC_NAMESPACE::open(TEST_DIR, O_DIRECTORY);
  ASSERT_GT(dir_fd, 0);
  ASSERT_EQ(libc_errno, 0);
  int file_fd = LIBC_NAMESPACE::open(TEST_FILE, O_PATH);
  ASSERT_GT(file_fd, 0);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(file_fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::fchdir(dir_fd), Succeeds(0));
  file_fd = LIBC_NAMESPACE::open(TEST_FILE_BASE, O_PATH);
  ASSERT_GT(file_fd, 0);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(file_fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(dir_fd), Succeeds(0));
}

TEST(LlvmLibcChdirTest, ChangeToNonExistentDir) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  libc_errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::fchdir(0), -1);
  ASSERT_NE(libc_errno, 0);
  libc_errno = 0;
}
