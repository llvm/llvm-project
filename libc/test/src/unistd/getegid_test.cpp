//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getegid.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/getegid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetEgidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetEgidTest, SmokeTest) {
  // getegid() always succeeds. Check that it returns a valid GID.
  ASSERT_GE(LIBC_NAMESPACE::getegid(), static_cast<gid_t>(0));
}
