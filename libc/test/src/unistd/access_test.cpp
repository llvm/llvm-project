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

#include <unistd.h>

TEST(LlvmLibcAccessTest, CreateAndTest) {
  // The test strategy is to repeatedly create a file in different modes and
  // test that it is accessable in those modes but not in others.
  libc_errno = 0;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE = "testdata/access.test";
  int fd = __llvm_libc::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  ASSERT_EQ(__llvm_libc::access(TEST_FILE, F_OK), 0);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_EQ(__llvm_libc::access(TEST_FILE, X_OK | W_OK | R_OK), 0);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_THAT(__llvm_libc::unlink(TEST_FILE), Succeeds(0));

  fd = __llvm_libc::open(TEST_FILE, O_WRONLY | O_CREAT, S_IXUSR);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));
  ASSERT_EQ(__llvm_libc::access(TEST_FILE, F_OK), 0);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_EQ(__llvm_libc::access(TEST_FILE, X_OK), 0);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_EQ(__llvm_libc::access(TEST_FILE, R_OK), -1);
  ASSERT_EQ(libc_errno, EACCES);
  libc_errno = 0;
  ASSERT_EQ(__llvm_libc::access(TEST_FILE, W_OK), -1);
  ASSERT_EQ(libc_errno, EACCES);
  libc_errno = 0;
  ASSERT_THAT(__llvm_libc::unlink(TEST_FILE), Succeeds(0));
}

TEST(LlvmLibcAccessTest, AccessNonExistentFile) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(__llvm_libc::access("testdata/non-existent-file", F_OK),
              Fails(ENOENT));
}
