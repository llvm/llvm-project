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
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGetrusageTest, BasicTest) {
  struct rusage usage = {};
  int res = LIBC_NAMESPACE::getrusage(RUSAGE_SELF, &usage);
  EXPECT_EQ(res, 0);
  ASSERT_ERRNO_SUCCESS();
}

TEST(LlvmLibcGetrusageTest, TestWhoInvalid) {
  struct rusage usage = {};
  int res = LIBC_NAMESPACE::getrusage(99, &usage);
  EXPECT_EQ(res, -1);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST(LlvmLibcGetrusageTest, TestUsageBadPointer) {
  int res = LIBC_NAMESPACE::getrusage(RUSAGE_SELF, (struct rusage *)-1L);
  EXPECT_EQ(res, -1);
  ASSERT_ERRNO_EQ(EFAULT);
}
