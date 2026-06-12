//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for __sched_cpualloc and __sched_cpufree.
///
//===----------------------------------------------------------------------===//

#include "hdr/sched_macros.h"
#include "hdr/types/cpu_set_t.h"
#include "include/llvm-libc-macros/linux/sched-macros.h"
#include "include/llvm-libc-types/cpu_set_t.h"
#include "src/sched/sched_cpualloc.h"
#include "src/sched/sched_cpufree.h"
#include "src/sched/sched_getcpucount.h"
#include "src/sched/sched_getcpuisset.h"
#include "src/sched/sched_setcpuset.h"
#include "src/sched/sched_setcpuzero.h"
#include "test/UnitTest/Test.h"

using LlvmLibcSchedCpuAllocTest = LIBC_NAMESPACE::testing::Test;

TEST_F(LlvmLibcSchedCpuAllocTest, AllocAndFreeSmall) {
  int num_cpus = 10;
  size_t alloc_size = CPU_ALLOC_SIZE(num_cpus);
  cpu_set_t *mask = LIBC_NAMESPACE::__sched_cpualloc(num_cpus);
  ASSERT_NE(mask, static_cast<cpu_set_t *>(nullptr));

  LIBC_NAMESPACE::__sched_setcpuzero(alloc_size, mask);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpucount(alloc_size, mask), 0);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpuisset(1, alloc_size, mask), 0);

  LIBC_NAMESPACE::__sched_setcpuset(1, alloc_size, mask);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpuisset(1, alloc_size, mask), 1);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpucount(alloc_size, mask), 1);

  LIBC_NAMESPACE::__sched_setcpuset(5, alloc_size, mask);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpuisset(5, alloc_size, mask), 1);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpucount(alloc_size, mask), 2);

  LIBC_NAMESPACE::__sched_cpufree(mask);
}

TEST_F(LlvmLibcSchedCpuAllocTest, AllocAndFreeLarge) {
  int num_cpus = 4096;
  size_t alloc_size = CPU_ALLOC_SIZE(num_cpus);
  cpu_set_t *mask = LIBC_NAMESPACE::__sched_cpualloc(num_cpus);
  ASSERT_NE(mask, static_cast<cpu_set_t *>(nullptr));

  LIBC_NAMESPACE::__sched_setcpuzero(alloc_size, mask);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpucount(alloc_size, mask), 0);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpuisset(1, alloc_size, mask), 0);

  LIBC_NAMESPACE::__sched_setcpuset(1, alloc_size, mask);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpuisset(1, alloc_size, mask), 1);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpucount(alloc_size, mask), 1);

  LIBC_NAMESPACE::__sched_setcpuset(500, alloc_size, mask);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpuisset(500, alloc_size, mask), 1);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpucount(alloc_size, mask), 2);

  LIBC_NAMESPACE::__sched_setcpuset(4095, alloc_size, mask);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpuisset(4095, alloc_size, mask), 1);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpucount(alloc_size, mask), 3);

  LIBC_NAMESPACE::__sched_cpufree(mask);
}

TEST_F(LlvmLibcSchedCpuAllocTest, AllocAndFreeZero) {
  cpu_set_t *mask = LIBC_NAMESPACE::__sched_cpualloc(/*count=*/0);
  LIBC_NAMESPACE::__sched_cpufree(mask);
}
