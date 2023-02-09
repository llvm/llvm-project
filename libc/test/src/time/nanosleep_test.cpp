//===-- Unittests for nanosleep
//---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <errno.h>
#include <time.h>

#include "src/time/nanosleep.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

namespace cpp = __llvm_libc::cpp;

TEST(LlvmLibcNanosleep, SmokeTest) {
  // TODO: When we have the code to read clocks, test that time has passed.
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  errno = 0;

  struct timespec tim = {1, 500};
  struct timespec tim2 = {0, 0};
  int ret = __llvm_libc::nanosleep(&tim, &tim2);
  ASSERT_EQ(errno, 0);
  ASSERT_EQ(ret, 0);
}
