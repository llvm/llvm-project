//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for setpriority.
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
using LlvmLibcSetpriorityTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSetpriorityTest,  BasicTest) {
  int nice = LIBC_NAMESPACE::getpriority(PRIO_PROCESS, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::setpriority(PRIO_PROCESS, 0, nice), Succeeds());
}

TEST_F(LlvmLibcSetpriorityTest, TestBadPid) {
  ASSERT_THAT(LIBC_NAMESPACE::setpriority(PRIO_PROCESS, -1, 19), Fails(ESRCH, -1));
}

TEST_F(LlvmLibcSetpriorityTest, TestBadWho) {
  ASSERT_THAT(LIBC_NAMESPACE::setpriority(99, 0, 19), Fails(EINVAL, -1));
}
