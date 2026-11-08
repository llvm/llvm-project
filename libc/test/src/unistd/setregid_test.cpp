//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setregid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/getegid.h"
#include "src/unistd/getgid.h"
#include "src/unistd/setregid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSetReGidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetReGidTest, NoopMinusOne) {
  // Passing -1 leaves the corresponding ID unchanged and always succeeds.
  ASSERT_THAT(
      LIBC_NAMESPACE::setregid(static_cast<gid_t>(-1), static_cast<gid_t>(-1)),
      Succeeds());
}

TEST_F(LlvmLibcSetReGidTest, SetCurrentReGid) {
  // Setting the real and effective GIDs to their current values should succeed.
  ASSERT_THAT(LIBC_NAMESPACE::setregid(LIBC_NAMESPACE::getgid(),
                                       LIBC_NAMESPACE::getegid()),
              Succeeds());
}

TEST_F(LlvmLibcSetReGidTest, InvalidGid) {
  ASSERT_THAT(
      LIBC_NAMESPACE::setregid(static_cast<gid_t>(-2), static_cast<gid_t>(-2)),
      Fails(any_of(EINVAL, EPERM)));
}
