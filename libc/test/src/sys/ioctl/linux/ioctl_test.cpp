//===-- Unittests for ioctl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/sys/ioctl/ioctl.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/sys_stat_macros.h"

#include "hdr/sys_ioctl_macros.h"

using LlvmLibcSysIoctlTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcSysIoctlTest, InvalidCommandAndFIONREAD) {
  // Setup the test file
  constexpr const char *TEST_FILE_NAME = "ioctl.test";
  constexpr const char TEST_MSG[] = "ioctl test";
  constexpr int TEST_MSG_SIZE = sizeof(TEST_MSG) - 1;
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int new_test_file_fd = LIBC_NAMESPACE::open(
      TEST_FILE, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  ASSERT_THAT(
      (int)LIBC_NAMESPACE::write(new_test_file_fd, TEST_MSG, TEST_MSG_SIZE),
      Succeeds(TEST_MSG_SIZE));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(new_test_file_fd), Succeeds(0));
  ASSERT_ERRNO_SUCCESS();

  // Reopen the file for testing
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

  // 0xDEADBEEF is just a random nonexistent command;
  // calling this should always fail with ENOTTY
  ret = LIBC_NAMESPACE::ioctl(fd, 0xDEADBEEF, NULL);
  ASSERT_ERRNO_EQ(ENOTTY);
  ASSERT_EQ(ret, -1);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}
