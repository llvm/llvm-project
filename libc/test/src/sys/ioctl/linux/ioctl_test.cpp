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

TEST(LlvmLibcSysIoctlTest, NullAndTestFileFIONREAD) {
  LIBC_NAMESPACE::libc_errno = 0;

  // FIONREAD reports the number of available bytes to read for the passed fd
  int dev_zero_fd = LIBC_NAMESPACE::open("/dev/zero", O_RDONLY);
  ASSERT_GT(dev_zero_fd, 0);
  ASSERT_ERRNO_SUCCESS();

  // For /dev/zero, this is always 0
  int dev_zero_n = -1;
  int ret = LIBC_NAMESPACE::ioctl(dev_zero_fd, FIONREAD, &dev_zero_n);
  ASSERT_GT(ret, -1);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(dev_zero_n, 0);

  ASSERT_THAT(LIBC_NAMESPACE::close(dev_zero_fd), Succeeds(0));

  // Now, with a file known to have a non-zero size
  constexpr const char TEST_MSG[] = "ioctl test";
  constexpr ssize_t TEST_MSG_SIZE = sizeof(TEST_MSG) - 1;
  constexpr const char *TEST_FILE = "testdata/ioctl.test";
  int test_file_fd = LIBC_NAMESPACE::open(TEST_FILE, O_RDONLY);
  ASSERT_GT(test_file_fd, 0);
  ASSERT_ERRNO_SUCCESS();

  // This reports the full size of the file, as we haven't read anything yet
  int test_file_n = -1;
  ret = LIBC_NAMESPACE::ioctl(test_file_fd, FIONREAD, &test_file_n);
  ASSERT_GT(ret, -1);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(test_file_n, TEST_MSG_SIZE);

  // But if we read some bytes...
  constexpr int READ_COUNT = 5;
  char buffer[READ_COUNT];
  ASSERT_THAT(LIBC_NAMESPACE::read(test_file_fd, buffer, READ_COUNT),
              Succeeds(READ_COUNT));

  // ... n should have decreased by the number of bytes we've read
  int test_file_n_after_reading = -1;
  ret =
      LIBC_NAMESPACE::ioctl(test_file_fd, FIONREAD, &test_file_n_after_reading);
  ASSERT_GT(ret, -1);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(test_file_n - READ_COUNT, test_file_n_after_reading);

  ASSERT_THAT(LIBC_NAMESPACE::close(test_file_fd), Succeeds(0));
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
