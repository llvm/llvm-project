//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getpriority.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_resource_macros.h"
#include "hdr/types/id_t.h"
#include "src/sys/resource/getpriority.h"
#include "src/sys/resource/setpriority.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcGetpriorityTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetpriorityTest, BasicTest) {
  int current_nice = LIBC_NAMESPACE::getpriority(PRIO_PROCESS, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(current_nice, -20);
  ASSERT_LE(current_nice, 19);

  // Increase to the max on Linux (i.e. minimal priority, which doesn't require
  // special privileges), so that we can confirm we get it back correctly.
  int nice = 19;
  // Make sure we're not setting it to the same priority by chance
  // however unlikely it may be.
  if (nice == current_nice) {
    nice -= 1;
  }
  ASSERT_THAT(LIBC_NAMESPACE::setpriority(PRIO_PROCESS, 0, nice), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::getpriority(PRIO_PROCESS, 0), Succeeds(nice));
}

TEST_F(LlvmLibcGetpriorityTest, TestBadPid) {
  ASSERT_THAT(LIBC_NAMESPACE::getpriority(PRIO_PROCESS, -1), Fails(ESRCH, -1));
}

TEST_F(LlvmLibcGetpriorityTest, TestBadWho) {
  ASSERT_THAT(LIBC_NAMESPACE::getpriority(99, 0), Fails(EINVAL, -1));
}
