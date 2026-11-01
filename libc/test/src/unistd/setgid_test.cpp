//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setgid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/getgid.h"
#include "src/unistd/setgid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSetGidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetGidTest, SetCurrentGid) {
  // Setting the GID to the current real GID is permitted and should succeed.
  ASSERT_THAT(LIBC_NAMESPACE::setgid(LIBC_NAMESPACE::getgid()), Succeeds());
}

TEST_F(LlvmLibcSetGidTest, InvalidGid) {
  ASSERT_THAT(LIBC_NAMESPACE::setgid(static_cast<gid_t>(-1)),
              Fails(any_of(EINVAL, EPERM)));
}
