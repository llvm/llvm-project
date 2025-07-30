//===-- Unittests for poll ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/limits_macros.h" // UINT_MAX
#include "src/__support/libc_errno.h"
#include "src/poll/poll.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcPollTest, SmokeTest) {
  libc_errno = 0;
  int ret = LIBC_NAMESPACE::poll(nullptr, 0, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(0, ret);
}
TEST(LlvmLibcPollTest, SmokeFailureTest) {
  libc_errno = 0;
  int ret = LIBC_NAMESPACE::poll(nullptr, UINT_MAX, 0);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_EQ(-1, ret);
}
