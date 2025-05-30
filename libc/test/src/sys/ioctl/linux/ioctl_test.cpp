//===-- Unittests for ioctl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/sys/ioctl/ioctl.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

#include <sys/ioctl.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcSysIoctlTest, TestFileFIONREAD) {
  LIBC_NAMESPACE::libc_errno = 0;

  constexpr const char TEST_MSG[] = "ioctl test";
  constexpr int TEST_MSG_SIZE = sizeof(TEST_MSG) - 1;
  constexpr const char *TEST_FILE = "testdata/ioctl.test";
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  // FIONREAD reports the number of available bytes to read for the passed fd
  // This will report the full size of the file, as we haven't read anything yet
  int n = -1;
  int ret = LIBC_NAMESPACE::ioctl(fd, FIONREAD, &n);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(ret, -1);
  ASSERT_EQ(n, TEST_MSG_SIZE);

  // But if we read some bytes...
  constexpr int READ_COUNT = 5;
  char read_buffer[READ_COUNT];
  ASSERT_THAT((int)LIBC_NAMESPACE::read(fd, read_buffer, READ_COUNT),
              Succeeds(READ_COUNT));

  // ... n should have decreased by the number of bytes we've read
  int n_after_reading = -1;
  ret = LIBC_NAMESPACE::ioctl(fd, FIONREAD, &n_after_reading);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(ret, -1);
  ASSERT_EQ(n - READ_COUNT, n_after_reading);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST(LlvmLibcSysIoctlTest, InvalidIoctlCommand) {
  LIBC_NAMESPACE::libc_errno = 0;

  int fd = LIBC_NAMESPACE::open("/dev/zero", O_RDONLY);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();

  // 0xDEADBEEF is just a random nonexistent command;
  // calling this should always fail with ENOTTY
  int ret = LIBC_NAMESPACE::ioctl(fd, 0xDEADBEEF, NULL);
  ASSERT_EQ(ret, -1);
  ASSERT_ERRNO_EQ(ENOTTY);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}
