//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setresuid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/geteuid.h"
#include "src/unistd/getuid.h"
#include "src/unistd/setresuid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSetResUidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetResUidTest, NoopMinusOne) {
  // Passing -1 leaves the corresponding ID unchanged and always succeeds.
  ASSERT_THAT(LIBC_NAMESPACE::setresuid(static_cast<uid_t>(-1),
                                        static_cast<uid_t>(-1),
                                        static_cast<uid_t>(-1)),
              Succeeds());
}

TEST_F(LlvmLibcSetResUidTest, SetCurrentResUid) {
  // Setting the real and effective UIDs to their current values should succeed.
  ASSERT_THAT(LIBC_NAMESPACE::setresuid(LIBC_NAMESPACE::getuid(),
                                        LIBC_NAMESPACE::geteuid(),
                                        static_cast<uid_t>(-1)),
              Succeeds());
}

TEST_F(LlvmLibcSetResUidTest, UnprivilegedInvalidUid) {
  if (LIBC_NAMESPACE::getuid() == 0)
    return;
  ASSERT_THAT(LIBC_NAMESPACE::setresuid(static_cast<uid_t>(-2),
                                        static_cast<uid_t>(-2),
                                        static_cast<uid_t>(-2)),
              Fails(any_of(EINVAL, EPERM)));
}
