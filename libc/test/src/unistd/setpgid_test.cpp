//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setpgid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/setpgid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSetPgidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetPgidTest, InvalidPid) {
  ASSERT_THAT(LIBC_NAMESPACE::setpgid(-1, 0), Fails(any_of(ESRCH, EINVAL)));
}

TEST_F(LlvmLibcSetPgidTest, InvalidPgid) {
  ASSERT_THAT(LIBC_NAMESPACE::setpgid(0, -1), Fails(EINVAL));
}

TEST_F(LlvmLibcSetPgidTest, SetpgidZero) {
  // setpgid(0, 0) sets the process group ID of the calling process to its PID.
  ASSERT_THAT(LIBC_NAMESPACE::setpgid(0, 0), Succeeds());
}
