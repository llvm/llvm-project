//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setegid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/getegid.h"
#include "src/unistd/getgid.h"
#include "src/unistd/setegid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSetEGidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetEGidTest, SetCurrentEGid) {
  // Setting the effective GID to the current effective GID should succeed.
  ASSERT_THAT(LIBC_NAMESPACE::setegid(LIBC_NAMESPACE::getegid()), Succeeds());
}

TEST_F(LlvmLibcSetEGidTest, SetCurrentGid) {
  // Setting the effective GID to the current real GID should succeed.
  ASSERT_THAT(LIBC_NAMESPACE::setegid(LIBC_NAMESPACE::getgid()), Succeeds());
}

TEST_F(LlvmLibcSetEGidTest, UnprivilegedInvalidGid) {
  if (LIBC_NAMESPACE::getgid() == 0)
    return;
  ASSERT_THAT(LIBC_NAMESPACE::setegid(static_cast<gid_t>(-2)),
              Fails(any_of(EINVAL, EPERM)));
}
