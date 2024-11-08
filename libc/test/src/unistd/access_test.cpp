//===-- Unittests for access ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/unistd/access.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/stat.h>
#include <unistd.h>

TEST(LlvmLibcAccessTest, CreateAndTest) {
  // The test strategy is to repeatedly create a file in different modes and
  // test that it is accessable in those modes but not in others.
  LIBC_NAMESPACE::libc_errno = 0;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "access.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_EQ(LIBC_NAMESPACE::access(TEST_FILE, F_OK), 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::access(TEST_FILE, X_OK | W_OK | R_OK), 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));

  fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IXUSR);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::access(TEST_FILE, F_OK), 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::access(TEST_FILE, X_OK), 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::access(TEST_FILE, R_OK), -1);
  ASSERT_ERRNO_EQ(EACCES);
  LIBC_NAMESPACE::libc_errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::access(TEST_FILE, W_OK), -1);
  ASSERT_ERRNO_EQ(EACCES);
  LIBC_NAMESPACE::libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
}

TEST(LlvmLibcAccessTest, AccessNonExistentFile) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::access("testdata/non-existent-file", F_OK),
              Fails(ENOENT));
}
