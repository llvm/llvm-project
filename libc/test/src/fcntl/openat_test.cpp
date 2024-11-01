//===-- Unittests for openat ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/fcntl/openat.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <fcntl.h>

TEST(LlvmLibcUniStd, OpenAndReadTest) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE = "openat.test";
  int dir_fd = __llvm_libc::open(TEST_DIR, O_DIRECTORY);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(dir_fd, 0);
  constexpr const char TEST_MSG[] = "openat test";
  constexpr int TEST_MSG_SIZE = sizeof(TEST_MSG) - 1;

  int read_fd = __llvm_libc::openat(dir_fd, TEST_FILE, O_RDONLY);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(read_fd, 0);
  char read_buf[TEST_MSG_SIZE];
  ASSERT_THAT(__llvm_libc::read(read_fd, read_buf, TEST_MSG_SIZE),
              Succeeds(TEST_MSG_SIZE));
  ASSERT_THAT(__llvm_libc::close(read_fd), Succeeds(0));
  ASSERT_THAT(__llvm_libc::close(dir_fd), Succeeds(0));
}

TEST(LlvmLibcUniStd, FailTest) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  EXPECT_THAT(__llvm_libc::openat(AT_FDCWD, "openat.test", O_RDONLY),
              Fails(ENOENT));
}
