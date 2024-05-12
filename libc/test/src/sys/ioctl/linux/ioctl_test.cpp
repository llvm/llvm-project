//===-- Unittests for ioctl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/errno/libc_errno.h"
#include "src/sys/ioctl/ioctl.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

#include <fcntl.h>
#include <sys/kd.h>
#include <sys/syscall.h>
#include <unistd.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcIoctlTest, InvalidFileDescriptor) {
  int fd = 10;
  unsigned long request = 10;
  int res = LIBC_NAMESPACE::ioctl(fd, request, NULL);
  EXPECT_THAT(res, Fails(EBADF, -1));
}

TEST(LlvmLibcIoctlTest, ValidFileDescriptor) {
  int fd = open("/dev/tty1", O_RDWR);
  int res = LIBC_NAMESPACE::ioctl(fd, KDSETMODE, KD_GRAPHICS);
  EXPECT_THAT(res, Succeeds());
}
