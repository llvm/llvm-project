//===-- Unittests for epoll_pwait -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/errno/libc_errno.h"
#include "src/sys/epoll/epoll_pwait.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcEpollWaitTest, Basic) {
  EXPECT_THAT(LIBC_NAMESPACE::epoll_pwait(-1, nullptr, 0, 0, nullptr),
              returns(EQ(-1ul)).with_errno(EQ(EINVAL)));
}

// TODO: Complete these tests when epoll_create is implemented.
