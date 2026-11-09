//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for flock.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_file_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/sys/file/flock.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFlockTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcFlockTest, LockAndUnlock) {
  constexpr const char *FILENAME = "flock.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  });

  // Acquire non-blocking shared lock.
  EXPECT_THAT(LIBC_NAMESPACE::flock(fd, LOCK_SH | LOCK_NB), Succeeds(0));
  // Release lock.
  EXPECT_THAT(LIBC_NAMESPACE::flock(fd, LOCK_UN), Succeeds(0));
}

TEST_F(LlvmLibcFlockTest, BadFd) {
  EXPECT_THAT(LIBC_NAMESPACE::flock(-1, LOCK_SH), Fails(EBADF));
}

TEST_F(LlvmLibcFlockTest, InvalidOp) {
  constexpr const char *FILENAME = "flock_invalid_op.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  });

  // Test various invalid op parameters.
  EXPECT_THAT(LIBC_NAMESPACE::flock(fd, 0), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::flock(fd, 0x1000), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::flock(fd, LOCK_SH | LOCK_EX), Fails(EINVAL));
}
