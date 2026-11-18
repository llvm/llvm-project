//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for unshare.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sched_macros.h"
#include "src/__support/libc_errno.h"
#include "src/sched/unshare.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcUnshareTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcUnshareTest, UnshareFs) {
  int ret = LIBC_NAMESPACE::unshare(CLONE_FS);
  if (ret == 0) {
    ASSERT_ERRNO_SUCCESS();
  } else {
    ASSERT_EQ(ret, -1);
    if (libc_errno == ENOSYS)
      ASSERT_ERRNO_EQ(ENOSYS);
    else
      ASSERT_ERRNO_EQ(EPERM);
  }
}

TEST_F(LlvmLibcUnshareTest, InvalidFlags) {
  int ret = LIBC_NAMESPACE::unshare(-1);
  ASSERT_EQ(ret, -1);
  if (libc_errno == ENOSYS)
    ASSERT_ERRNO_EQ(ENOSYS);
  else if (libc_errno == EPERM)
    ASSERT_ERRNO_EQ(EPERM);
  else
    ASSERT_ERRNO_EQ(EINVAL);
}
