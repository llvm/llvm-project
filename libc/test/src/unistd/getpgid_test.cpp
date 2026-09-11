//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getpgid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/getpgid.h"
#include "src/unistd/getpid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcGetPgidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetPgidTest, GetCurrPGID) {
  pid_t pgid = LIBC_NAMESPACE::getpgid(0);
  ASSERT_GT(pgid, 0);
}

TEST_F(LlvmLibcGetPgidTest, GetSelfPGID) {
  pid_t pgid_zero = LIBC_NAMESPACE::getpgid(0);
  pid_t pgid_self = LIBC_NAMESPACE::getpgid(LIBC_NAMESPACE::getpid());
  ASSERT_GT(pgid_zero, 0);
  ASSERT_EQ(pgid_zero, pgid_self);
}

TEST_F(LlvmLibcGetPgidTest, InvalidPID) {
  ASSERT_THAT(LIBC_NAMESPACE::getpgid(-1), Fails<pid_t>(any_of(ESRCH, EINVAL)));
}
