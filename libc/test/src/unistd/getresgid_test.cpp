//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getresgid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/getegid.h"
#include "src/unistd/getgid.h"
#include "src/unistd/getresgid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcGetResGidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetResGidTest, GetCurrentResGid) {
  gid_t rgid = static_cast<gid_t>(-1);
  gid_t egid = static_cast<gid_t>(-1);
  gid_t sgid = static_cast<gid_t>(-1);
  ASSERT_THAT(LIBC_NAMESPACE::getresgid(&rgid, &egid, &sgid), Succeeds());
  EXPECT_EQ(rgid, LIBC_NAMESPACE::getgid());
  EXPECT_EQ(egid, LIBC_NAMESPACE::getegid());
  EXPECT_NE(sgid, static_cast<gid_t>(-1));
}

TEST_F(LlvmLibcGetResGidTest, InvalidPointer) {
  ASSERT_THAT(LIBC_NAMESPACE::getresgid(nullptr, nullptr, nullptr),
              Fails(EFAULT));
}
