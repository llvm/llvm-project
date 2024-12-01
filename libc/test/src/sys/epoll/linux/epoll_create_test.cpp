//===-- Unittests for epoll_create ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/errno/libc_errno.h"
#include "src/sys/epoll/epoll_create.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include <sys/syscall.h> // For syscall numbers.

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcEpollCreateTest, Basic) {
  int fd = LIBC_NAMESPACE::epoll_create(1);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds());
}

#ifdef SYS_epoll_create
TEST(LlvmLibcEpollCreateTest, Fails) {
  ASSERT_THAT(LIBC_NAMESPACE::epoll_create(0), Fails(EINVAL));
}
#endif
