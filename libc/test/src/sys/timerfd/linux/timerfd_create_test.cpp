//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for timerfd_create.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_timerfd_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/fcntl.h"
#include "src/sys/timerfd/timerfd_create.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LIBC_NAMESPACE::cpp::scope_exit;
using LlvmLibcTimerfdCreateTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcTimerfdCreateTest, ZeroFlags) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::timerfd_create(CLOCK_MONOTONIC, 0),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit close_fd(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0)); });

  int flags;
  ASSERT_THAT(flags = LIBC_NAMESPACE::fcntl(fd, F_GETFD),
              returns(GE(0)).with_errno(EQ(0)));
  EXPECT_EQ(flags & FD_CLOEXEC, 0);

  ASSERT_THAT(flags = LIBC_NAMESPACE::fcntl(fd, F_GETFL),
              returns(GE(0)).with_errno(EQ(0)));
  EXPECT_EQ(flags & O_NONBLOCK, 0);
}

TEST_F(LlvmLibcTimerfdCreateTest, CloexecAndNonblockRead) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::timerfd_create(CLOCK_REALTIME,
                                                  TFD_CLOEXEC | TFD_NONBLOCK),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit close_fd(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0)); });

  int flags;
  ASSERT_THAT(flags = LIBC_NAMESPACE::fcntl(fd, F_GETFD),
              returns(GE(0)).with_errno(EQ(0)));
  EXPECT_NE(flags & FD_CLOEXEC, 0);

  ASSERT_THAT(flags = LIBC_NAMESPACE::fcntl(fd, F_GETFL),
              returns(GE(0)).with_errno(EQ(0)));
  EXPECT_NE(flags & O_NONBLOCK, 0);

  uint64_t val = 0;
  // Reading smaller than sizeof(uint64_t) from a timerfd fails with EINVAL.
  EXPECT_THAT(LIBC_NAMESPACE::read(fd, &val, sizeof(val) - 1),
              Fails<ssize_t>(EINVAL));
  // Disarmed non-blocking timerfd has no expirations and fails with EAGAIN
  // instead of blocking.
  EXPECT_THAT(LIBC_NAMESPACE::read(fd, &val, sizeof(val)),
              Fails<ssize_t>(EAGAIN));
}

TEST_F(LlvmLibcTimerfdCreateTest, InvalidArgs) {
  EXPECT_THAT(LIBC_NAMESPACE::timerfd_create(-1, 0), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::timerfd_create(CLOCK_MONOTONIC, -1),
              Fails(EINVAL));
}
