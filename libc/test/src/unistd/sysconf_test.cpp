//===-- Unittests for sysconf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/sysconf.h"
#include "test/UnitTest/Test.h"

#include <sys/sysinfo.h>
#include <unistd.h>

TEST(LlvmLibcSysconfTest, PagesizeTest) {
  long pagesize = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);
  ASSERT_GT(pagesize, 0l);
}

TEST(LlvmLibcSysconfTest, NprocessorsConfTest) {
  int cpu_count = ::get_nprocs_conf();
  ASSERT_GT(cpu_count, 0);

  long sysconf_count = LIBC_NAMESPACE::sysconf(_SC_NPROCESSORS_CONF);
  ASSERT_GT(sysconf_count, 0l);
  EXPECT_EQ(sysconf_count, static_cast<long>(cpu_count));
}

TEST(LlvmLibcSysconfTest, NprocessorsOnlnTest) {
  int cpu_count = ::get_nprocs();
  ASSERT_GT(cpu_count, 0);

  long sysconf_count = LIBC_NAMESPACE::sysconf(_SC_NPROCESSORS_ONLN);
  ASSERT_GT(sysconf_count, 0l);
  EXPECT_EQ(sysconf_count, static_cast<long>(cpu_count));
}
