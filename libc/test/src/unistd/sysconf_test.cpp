//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sysconf
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/unistd_macros.h"
#include "src/unistd/sysconf.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSysconfTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSysconfTest, PagesizeTest) {
  long pagesize = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);
  ASSERT_GT(pagesize, 0L);
}

TEST_F(LlvmLibcSysconfTest, NprocessorsConfTest) {
  long sysconf_count = LIBC_NAMESPACE::sysconf(_SC_NPROCESSORS_CONF);
  ASSERT_GT(sysconf_count, 0L);
}

TEST_F(LlvmLibcSysconfTest, NprocessorsOnlnTest) {
  long sysconf_count = LIBC_NAMESPACE::sysconf(_SC_NPROCESSORS_ONLN);
  ASSERT_GT(sysconf_count, 0L);
}

TEST_F(LlvmLibcSysconfTest, ThreadsTest) {
  long threads = LIBC_NAMESPACE::sysconf(_SC_THREADS);
  ASSERT_EQ(threads, _POSIX_THREADS);
}

TEST_F(LlvmLibcSysconfTest, ArgMaxTest) {
  long arg_max = LIBC_NAMESPACE::sysconf(_SC_ARG_MAX);
  ASSERT_GT(arg_max, 0L);
  ASSERT_GE(arg_max, 131072L);
}

TEST_F(LlvmLibcSysconfTest, OpenMaxTest) {
  long open_max = LIBC_NAMESPACE::sysconf(_SC_OPEN_MAX);
  if (open_max == -1)
    return;
  ASSERT_GT(open_max, 0L);
}

TEST_F(LlvmLibcSysconfTest, PhysPagesTest) {
  long phys_pages = LIBC_NAMESPACE::sysconf(_SC_PHYS_PAGES);
  ASSERT_GT(phys_pages, 0L);
}

TEST_F(LlvmLibcSysconfTest, KnownConstantValuesTest) {
  EXPECT_EQ(LIBC_NAMESPACE::sysconf(_SC_CLK_TCK), 100L);
  EXPECT_EQ(LIBC_NAMESPACE::sysconf(_SC_GETGR_R_SIZE_MAX), -1L);
  EXPECT_EQ(LIBC_NAMESPACE::sysconf(_SC_GETPW_R_SIZE_MAX), -1L);
}

TEST_F(LlvmLibcSysconfTest, InvalidNameTest) {
  EXPECT_THAT(LIBC_NAMESPACE::sysconf(100000), Fails(EINVAL, -1L));
  EXPECT_THAT(LIBC_NAMESPACE::sysconf(0x7fffffff), Fails(EINVAL, -1L));
}
