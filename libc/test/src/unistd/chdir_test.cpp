//===-- Unittests for chdir -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/unistd/chdir.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <fcntl.h>

TEST(LlvmLibcChdirTest, ChangeAndOpen) {
  // The idea of this test is that we will first open an existing test file
  // without changing the directory to make sure it exists. Next, we change
  // directory and open the same file to make sure that the "chdir" operation
  // succeeded.
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE = "testdata/chdir.test";
  constexpr const char *TEST_FILE_BASE = "chdir.test";
  libc_errno = 0;

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_PATH);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::chdir(TEST_DIR), Succeeds(0));
  fd = LIBC_NAMESPACE::open(TEST_FILE_BASE, O_PATH);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST(LlvmLibcChdirTest, ChangeToNonExistentDir) {
  libc_errno = 0;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::chdir("non-existent-dir"), Fails(ENOENT));
  libc_errno = 0;
}
