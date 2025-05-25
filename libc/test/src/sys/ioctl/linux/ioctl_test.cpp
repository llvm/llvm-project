//===-- Unittests for ioctl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sys/ioctl/ioctl.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include <sys/filio.h>
#include <sys/ioctl.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcSysIoctlTest, StdinFIONREAD) {
  LIBC_NAMESPACE::libc_errno = 0;

  // FIONREAD reports the number of readable bytes for fd
  int bytes;
  int ret = LIBC_NAMESPACE::ioctl(0, FIONREAD, &bytes);
  ASSERT_ERRNO_SUCCESS();
}

TEST(LlvmLibcSysIoctlTest, InvalidCommandENOTTY) {
  LIBC_NAMESPACE::libc_errno = 0;

  // 0xDEADBEEF is just a random nonexistent command;
  // calling this should always fail with ENOTTY
  int ret = LIBC_NAMESPACE::ioctl(3, 0xDEADBEEF, NULL);
  ASSERT_TRUE(ret == -1 && errno == ENOTTY);
}
