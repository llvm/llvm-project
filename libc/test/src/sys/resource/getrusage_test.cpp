//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getrusage.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_resource_macros.h"
#include "hdr/types/struct_rusage.h"
#include "src/sys/resource/getrusage.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcGetrusageTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetrusageTest, BasicTest) {
  struct rusage usage = {};
  ASSERT_THAT(LIBC_NAMESPACE::getrusage(RUSAGE_SELF, &usage), Succeeds());
  // The number of soft page faults should be stably greater than 0.
  ASSERT_TRUE(usage.ru_minflt > 0);
}

TEST_F(LlvmLibcGetrusageTest, TestWhoInvalid) {
  struct rusage usage = {};
  ASSERT_THAT(LIBC_NAMESPACE::getrusage(99, &usage), Fails(EINVAL, -1));
}

TEST_F(LlvmLibcGetrusageTest, TestUsageBadPointer) {
  ASSERT_THAT(LIBC_NAMESPACE::getrusage(RUSAGE_SELF, (struct rusage *)-1L), Fails(EFAULT, -1));
}
