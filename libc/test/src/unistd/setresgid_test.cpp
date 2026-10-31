//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setresgid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/getegid.h"
#include "src/unistd/getgid.h"
#include "src/unistd/setresgid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSetResGidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetResGidTest, NoopMinusOne) {
  // Passing -1 leaves the corresponding ID unchanged and always succeeds.
  ASSERT_THAT(LIBC_NAMESPACE::setresgid(static_cast<gid_t>(-1),
                                        static_cast<gid_t>(-1),
                                        static_cast<gid_t>(-1)),
              Succeeds());
}

TEST_F(LlvmLibcSetResGidTest, SetCurrentResGid) {
  // Setting the real and effective GIDs to their current values should succeed.
  ASSERT_THAT(LIBC_NAMESPACE::setresgid(LIBC_NAMESPACE::getgid(),
                                        LIBC_NAMESPACE::getegid(),
                                        static_cast<gid_t>(-1)),
              Succeeds());
}

TEST_F(LlvmLibcSetResGidTest, UnprivilegedInvalidGid) {
  if (LIBC_NAMESPACE::getgid() == 0)
    return;
  ASSERT_THAT(LIBC_NAMESPACE::setresgid(static_cast<gid_t>(-2),
                                        static_cast<gid_t>(-2),
                                        static_cast<gid_t>(-2)),
              Fails(any_of(EINVAL, EPERM)));
}
