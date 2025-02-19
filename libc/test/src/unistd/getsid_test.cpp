//===-- Unittests for getsid ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/unistd/getsid.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGetPidTest, GetCurrSID) {
  pid_t sid = LIBC_NAMESPACE::getsid(0);
  ASSERT_NE(sid, -1);
  ASSERT_ERRNO_SUCCESS();

  pid_t nonexist_sid = LIBC_NAMESPACE::getsid(-1);
  ASSERT_EQ(nonexist_sid, -1);
  ASSERT_ERRNO_FAILURE();
}
