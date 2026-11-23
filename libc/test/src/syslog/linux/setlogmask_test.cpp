//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setlogmask.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/syslog_macros.h"
#include "src/__support/libc_errno.h"
#include "src/syslog/setlogmask.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcSetlogmaskTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetlogmaskTest, DefaultAllowsAllPriorities) {
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(0), LOG_UPTO(LOG_DEBUG));
}

TEST_F(LlvmLibcSetlogmaskTest, RoundTrip) {
  int original = LIBC_NAMESPACE::setlogmask(LOG_UPTO(LOG_ERR));
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(LOG_MASK(LOG_INFO)), LOG_UPTO(LOG_ERR));
  int combined = LOG_MASK(LOG_EMERG) | LOG_MASK(LOG_INFO);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(combined), LOG_MASK(LOG_INFO));
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(original), combined);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(0), original);
}

TEST_F(LlvmLibcSetlogmaskTest, ZeroLeavesMaskUnchanged) {
  int original = LIBC_NAMESPACE::setlogmask(0);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(0), original);

  int custom = LOG_MASK(LOG_ERR);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(custom), original);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(0), custom);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(0), custom);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(original), custom);
}

TEST_F(LlvmLibcSetlogmaskTest, PreservesErrno) {
  libc_errno = EINVAL;
  int original = LIBC_NAMESPACE::setlogmask(LOG_UPTO(LOG_ERR));
  ASSERT_EQ(static_cast<int>(libc_errno), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(0), LOG_UPTO(LOG_ERR));
  ASSERT_EQ(static_cast<int>(libc_errno), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(original), LOG_UPTO(LOG_ERR));
  ASSERT_ERRNO_EQ(EINVAL);
}
