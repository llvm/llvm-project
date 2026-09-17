//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for setgroups.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "src/grp/setgroups.h"
#include "src/unistd/getuid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::any_of;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSetgroupsTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetgroupsTest, InvalidSizeReturnsEinval) {
  gid_t list[1] = {0};
  EXPECT_THAT(LIBC_NAMESPACE::setgroups(static_cast<size_t>(-1), list),
              Fails(any_of(EINVAL, EPERM)));
}

TEST_F(LlvmLibcSetgroupsTest, PrivilegeCheck) {
  gid_t list[1] = {1000};
  if (LIBC_NAMESPACE::getuid() == 0) {
    EXPECT_THAT(LIBC_NAMESPACE::setgroups(1, list), Succeeds(0));
  } else {
    EXPECT_THAT(LIBC_NAMESPACE::setgroups(1, list), Fails(EPERM));
  }
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcSetgroupsTest, NullPointerCrash) {
  ASSERT_DEATH([] { LIBC_NAMESPACE::setgroups(1, nullptr); }, WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
