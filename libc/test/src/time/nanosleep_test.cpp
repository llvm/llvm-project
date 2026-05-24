//===-- Unittests for nanosleep -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_timespec.h"
#include "src/time/nanosleep.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

namespace cpp = LIBC_NAMESPACE::cpp;

using LlvmLibcNanosleep = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcNanosleep, SmokeTest) {
  // TODO: When we have the code to read clocks, test that time has passed.
  struct timespec tim = {1, 500};
  struct timespec tim2 = {0, 0};
  ASSERT_EQ(LIBC_NAMESPACE::nanosleep(&tim, &tim2), 0);
}
