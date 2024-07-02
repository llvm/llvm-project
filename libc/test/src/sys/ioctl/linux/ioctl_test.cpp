//===-- Unittests for ioctl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/sys/ioctl/ioctl.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

#include <linux/fs.h>

TEST(LlvmLibcIoctlTest, InvalidFileDescriptor) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  int fd = 10;
  unsigned long request = 10;
  int res = LIBC_NAMESPACE::ioctl(fd, request, NULL);
  EXPECT_THAT(res, Fails(EBADF, -1));
}

TEST(LlvmLibcIoctlTest, ValidFileDescriptor) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  LIBC_NAMESPACE::libc_errno = 0;
  constexpr const char *FILENAME = "ioctl.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  int data;
  int res = LIBC_NAMESPACE::ioctl(fd, FS_IOC_GETFLAGS, &data);
  EXPECT_THAT(res, Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}
