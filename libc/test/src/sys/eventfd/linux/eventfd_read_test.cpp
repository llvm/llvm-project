//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for eventfd_read.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_eventfd_macros.h"
#include "hdr/types/eventfd_t.h"
#include "src/sys/eventfd/eventfd.h"
#include "src/sys/eventfd/eventfd_read.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcEventfdReadTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcEventfdReadTest, SecondReadFromNonblockReturnsEagain) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::eventfd(42, EFD_NONBLOCK),
              returns(GE(0)).with_errno(EQ(0)));

  eventfd_t val = 0;
  ASSERT_THAT(LIBC_NAMESPACE::eventfd_read(fd, &val), Succeeds(0));
  EXPECT_EQ(val, static_cast<eventfd_t>(42));

  ASSERT_THAT(LIBC_NAMESPACE::eventfd_read(fd, &val), Fails(EAGAIN));

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcEventfdReadTest, InvalidFd) {
  eventfd_t val = 0;
  ASSERT_THAT(LIBC_NAMESPACE::eventfd_read(-1, &val), Fails(EBADF));
}
