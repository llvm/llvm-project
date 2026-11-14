//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for eventfd_write.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_eventfd_macros.h"
#include "src/sys/eventfd/eventfd.h"
#include "src/sys/eventfd/eventfd_write.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcEventfdWriteTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcEventfdWriteTest, WritesAccumulateValues) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::eventfd(0, EFD_NONBLOCK),
              returns(GE(0)).with_errno(EQ(0)));

  ASSERT_THAT(LIBC_NAMESPACE::eventfd_write(fd, 10), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::eventfd_write(fd, 32), Succeeds(0));

  uint64_t val = 0;
  ASSERT_THAT(LIBC_NAMESPACE::read(fd, &val, sizeof(val)),
              Succeeds<ssize_t>(sizeof(val)));
  EXPECT_EQ(val, static_cast<uint64_t>(42));

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcEventfdWriteTest, Uint64MaxWriteFails) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::eventfd(0, 0),
              returns(GE(0)).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::eventfd_write(fd, static_cast<uint64_t>(-1)),
              Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcEventfdWriteTest, InvalidFd) {
  ASSERT_THAT(LIBC_NAMESPACE::eventfd_write(-1, 1), Fails(EBADF));
}
