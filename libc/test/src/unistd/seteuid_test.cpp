//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for seteuid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/geteuid.h"
#include "src/unistd/getuid.h"
#include "src/unistd/seteuid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSetEUidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetEUidTest, SetCurrentEUid) {
  // Setting the effective UID to the current effective UID should succeed.
  ASSERT_THAT(LIBC_NAMESPACE::seteuid(LIBC_NAMESPACE::geteuid()), Succeeds());
}

TEST_F(LlvmLibcSetEUidTest, SetCurrentUid) {
  // Setting the effective UID to the current real UID should succeed.
  ASSERT_THAT(LIBC_NAMESPACE::seteuid(LIBC_NAMESPACE::getuid()), Succeeds());
}

TEST_F(LlvmLibcSetEUidTest, UnprivilegedInvalidUid) {
  if (LIBC_NAMESPACE::getuid() == 0)
    return;
  ASSERT_THAT(LIBC_NAMESPACE::seteuid(static_cast<uid_t>(-2)),
              Fails(any_of(EINVAL, EPERM)));
}
