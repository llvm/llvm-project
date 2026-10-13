//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittest for lockf.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/unistd_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/lockf.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLockfTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcLockfTest, BadFd) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  EXPECT_THAT(LIBC_NAMESPACE::lockf(-1, F_LOCK, 0), Fails(EBADF));
  EXPECT_THAT(LIBC_NAMESPACE::lockf(-1, F_TLOCK, 0), Fails(EBADF));
  EXPECT_THAT(LIBC_NAMESPACE::lockf(-1, F_ULOCK, 0), Fails(EBADF));
  EXPECT_THAT(LIBC_NAMESPACE::lockf(-1, F_TEST, 0), Fails(EBADF));
}

TEST_F(LlvmLibcLockfTest, InvalidCmd) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/lockf_invalid_cmd.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  });

  EXPECT_THAT(LIBC_NAMESPACE::lockf(fd, 999, 0), Fails(EINVAL));
}

TEST_F(LlvmLibcLockfTest, LockUnlockAndTest) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/lockf_basic.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  });

  // Section is not locked: F_TEST should succeed (return 0).
  EXPECT_THAT(LIBC_NAMESPACE::lockf(fd, F_TEST, 0), Succeeds(0));

  // Lock the entire file: F_LOCK should succeed.
  EXPECT_THAT(LIBC_NAMESPACE::lockf(fd, F_LOCK, 0), Succeeds(0));

  // When locked by calling process, F_TEST still succeeds (POSIX requirement).
  EXPECT_THAT(LIBC_NAMESPACE::lockf(fd, F_TEST, 0), Succeeds(0));

  // Unlock the entire file: F_ULOCK should succeed.
  EXPECT_THAT(LIBC_NAMESPACE::lockf(fd, F_ULOCK, 0), Succeeds(0));

  // Non-blocking lock: F_TLOCK should succeed.
  EXPECT_THAT(LIBC_NAMESPACE::lockf(fd, F_TLOCK, 100), Succeeds(0));

  // Unlock the section: F_ULOCK should succeed.
  EXPECT_THAT(LIBC_NAMESPACE::lockf(fd, F_ULOCK, 100), Succeeds(0));
}

TEST_F(LlvmLibcLockfTest, ReadOnlyFd) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/lockf_rdonly.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_RDONLY, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  });

  // F_LOCK on a read-only fd should fail with EBADF.
  EXPECT_THAT(LIBC_NAMESPACE::lockf(fd, F_LOCK, 0), Fails(EBADF));
  EXPECT_THAT(LIBC_NAMESPACE::lockf(fd, F_TLOCK, 0), Fails(EBADF));
}
