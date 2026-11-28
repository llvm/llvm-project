//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for klogctl.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/libc_errno.h"
#include "src/sys/klog/klogctl.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcKlogctlTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcKlogctlTest, SizeBuffer) {
  constexpr int SYSLOG_ACTION_SIZE_BUFFER = 10;
  int ret = LIBC_NAMESPACE::klogctl(SYSLOG_ACTION_SIZE_BUFFER, nullptr, 0);
  if (ret >= 0) {
    ASSERT_ERRNO_SUCCESS();
  } else {
    ASSERT_EQ(ret, -1);
    if (libc_errno == ENOSYS)
      ASSERT_ERRNO_EQ(ENOSYS);
    else
      ASSERT_ERRNO_EQ(EPERM);
  }
}

TEST_F(LlvmLibcKlogctlTest, InvalidAction) {
  constexpr int INVALID_SYSLOG_ACTION = -1;
  int ret = LIBC_NAMESPACE::klogctl(INVALID_SYSLOG_ACTION, nullptr, 0);
  ASSERT_EQ(ret, -1);
  if (libc_errno == EPERM)
    ASSERT_ERRNO_EQ(EPERM);
  else if (libc_errno == ENOSYS)
    ASSERT_ERRNO_EQ(ENOSYS);
  else
    ASSERT_ERRNO_EQ(EINVAL);
}
