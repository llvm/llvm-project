//===-- Unittests for nanosleep -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <time.h>

#include "src/errno/libc_errno.h"
#include "src/time/nanosleep.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

namespace cpp = LIBC_NAMESPACE::cpp;

TEST(LlvmLibcNanosleep, SmokeTest) {
  // TODO: When we have the code to read clocks, test that time has passed.
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  LIBC_NAMESPACE::libc_errno = 0;

  struct timespec tim = {1, 500};
  struct timespec tim2 = {0, 0};
  int ret = LIBC_NAMESPACE::nanosleep(&tim, &tim2);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(ret, 0);
}
