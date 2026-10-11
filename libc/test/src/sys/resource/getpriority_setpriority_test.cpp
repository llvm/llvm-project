//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getpriority and setpriority.
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
using LlvmLibcGetprioritySetpriorityTest =
    LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetprioritySetpriorityTest, BesicTest) {
  int current_nice = LIBC_NAMESPACE::getpriority(PRIO_PROCESS, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(current_nice, -20);
  ASSERT_LE(current_nice, 19);

  // 19 is the highest possible nice on Linux (i.e. minimal priority), which
  // doesn't require special privileges. Current nice is likely less than or at
  // worst equal to 19. This allows us to round-trip this value and confirm
  // that we get back the correct nice.
  int nice = 19;

  ASSERT_THAT(LIBC_NAMESPACE::setpriority(PRIO_PROCESS, 0, nice), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::getpriority(PRIO_PROCESS, 0), Succeeds(nice));
}

TEST_F(LlvmLibcGetprioritySetpriorityTest, TestBadPidGetpriority) {
  ASSERT_THAT(LIBC_NAMESPACE::getpriority(PRIO_PROCESS, -1), Fails(ESRCH, -1));
}

TEST_F(LlvmLibcGetprioritySetpriorityTest, TestBadWhichGetpriority) {
  ASSERT_THAT(LIBC_NAMESPACE::getpriority(99, 0), Fails(EINVAL, -1));
}

TEST_F(LlvmLibcGetprioritySetpriorityTest, TestBadPidSetpriority) {
  ASSERT_THAT(LIBC_NAMESPACE::setpriority(PRIO_PROCESS, -1, 19),
              Fails(ESRCH, -1));
}

TEST_F(LlvmLibcGetprioritySetpriorityTest, TestBadWhichSetpriority) {
  ASSERT_THAT(LIBC_NAMESPACE::setpriority(99, 0, 19), Fails(EINVAL, -1));
}
