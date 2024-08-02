//===-- Unittests for epoll_ctl -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_epoll_macros.h"
#include "hdr/types/struct_epoll_event.h"
#include "src/errno/libc_errno.h"
#include "src/sys/epoll/epoll_create1.h"
#include "src/sys/epoll/epoll_ctl.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcEpollCtlTest, Basic) {
  int epfd = LIBC_NAMESPACE::epoll_create1(0);
  ASSERT_GT(epfd, 0);
  ASSERT_ERRNO_SUCCESS();

  int pipefd[2];

  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds());

  epoll_event event;
  event.events = EPOLLOUT;
  event.data.fd = pipefd[0];

  ASSERT_THAT(LIBC_NAMESPACE::epoll_ctl(epfd, EPOLL_CTL_ADD, pipefd[0], &event),
              Succeeds());

  // adding the same file fail.
  ASSERT_THAT(LIBC_NAMESPACE::epoll_ctl(epfd, EPOLL_CTL_ADD, pipefd[0], &event),
              Fails(EEXIST));

  ASSERT_THAT(LIBC_NAMESPACE::epoll_ctl(epfd, EPOLL_CTL_DEL, pipefd[0], &event),
              Succeeds());

  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::close(epfd), Succeeds());
}
