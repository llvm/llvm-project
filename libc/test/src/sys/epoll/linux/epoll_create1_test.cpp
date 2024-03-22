//===-- Unittests for epoll_create1 ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/linux/sys-epoll-macros.h"
#include "src/errno/libc_errno.h"
#include "src/sys/epoll/epoll_create1.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcEpollCreate1Test, Basic) {
  int fd = LIBC_NAMESPACE::epoll_create1(0);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds());
}

TEST(LlvmLibcEpollCreate1Test, CloseOnExecute) {
  int fd = LIBC_NAMESPACE::epoll_create1(EPOLL_CLOEXEC);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds());
}
