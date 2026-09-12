//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getpgrp.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/getpgid.h"
#include "src/unistd/getpgrp.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetPgrpTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetPgrpTest, SmokeTest) {
  // getpgrp() always succeeds. Simply check that it returns a sane value.
  pid_t pgrp = LIBC_NAMESPACE::getpgrp();
  ASSERT_GT(pgrp, 0);
}

TEST_F(LlvmLibcGetPgrpTest, MatchesGetPgidZero) {
  // getpgrp() is equivalent to getpgid(0).
  ASSERT_EQ(LIBC_NAMESPACE::getpgrp(), LIBC_NAMESPACE::getpgid(0));
}
