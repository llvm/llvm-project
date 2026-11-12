//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for eventfd.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_eventfd_macros.h"
#include "src/fcntl/fcntl.h"
#include "src/sys/eventfd/eventfd.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcEventfdTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcEventfdTest, ZeroFlags) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::eventfd(42, 0),
              returns(GE(0)).with_errno(EQ(0)));
  uint64_t val = 0;
  ASSERT_THAT(LIBC_NAMESPACE::read(fd, &val, sizeof(val)),
              Succeeds<ssize_t>(sizeof(val)));
  EXPECT_EQ(val, static_cast<uint64_t>(42));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcEventfdTest, EfdSemaphore) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::eventfd(42, EFD_CLOEXEC | EFD_NONBLOCK |
                                                   EFD_SEMAPHORE),
              returns(GE(0)).with_errno(EQ(0)));

  // Verify that returned descriptor has CLOEXEC and NONBLOCK flags set.
  int flags;
  ASSERT_THAT(flags = LIBC_NAMESPACE::fcntl(fd, F_GETFD),
              returns(GE(0)).with_errno(EQ(0)));
  EXPECT_NE(flags & FD_CLOEXEC, 0);

  ASSERT_THAT(flags = LIBC_NAMESPACE::fcntl(fd, F_GETFL),
              returns(GE(0)).with_errno(EQ(0)));
  EXPECT_NE(flags & O_NONBLOCK, 0);

  uint64_t val = 0;
  ASSERT_THAT(LIBC_NAMESPACE::read(fd, &val, sizeof(val)),
              Succeeds<ssize_t>(sizeof(val)));
  EXPECT_EQ(val, static_cast<uint64_t>(1));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcEventfdTest, InvalidFlags) {
  ASSERT_THAT(LIBC_NAMESPACE::eventfd(0, -1), Fails(EINVAL));
}
