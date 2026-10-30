//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getgroups.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/gid_t.h"
#include "src/unistd/getgroups.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcGetGroupsTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetGroupsTest, GetCurrentGroups) {
  int count = LIBC_NAMESPACE::getgroups(0, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(count, 0);

  gid_t groups[64];
  if (count <= 64) {
    EXPECT_THAT(LIBC_NAMESPACE::getgroups(64, groups), Succeeds(count));
  }
}

TEST_F(LlvmLibcGetGroupsTest, NegativeSizeReturnsEinval) {
  gid_t list[1] = {0};
  EXPECT_THAT(LIBC_NAMESPACE::getgroups(-1, list), Fails(EINVAL));
}

TEST_F(LlvmLibcGetGroupsTest, NullptrWithNonZeroSize) {
  int count = LIBC_NAMESPACE::getgroups(0, nullptr);
  ASSERT_ERRNO_SUCCESS();
  if (count > 0) {
    EXPECT_THAT(LIBC_NAMESPACE::getgroups(count, nullptr), Fails(EFAULT));
  }
}
