//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setuid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/getuid.h"
#include "src/unistd/setuid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSetUidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetUidTest, SetCurrentUid) {
  // Setting the UID to the current real UID is permitted and should succeed.
  ASSERT_THAT(LIBC_NAMESPACE::setuid(LIBC_NAMESPACE::getuid()), Succeeds());
}

TEST_F(LlvmLibcSetUidTest, InvalidUid) {
  // Setting an invalid UID should fail with EINVAL or EPERM.
  ASSERT_THAT(LIBC_NAMESPACE::setuid(static_cast<uid_t>(-1)),
              Fails(any_of(EINVAL, EPERM)));
}
