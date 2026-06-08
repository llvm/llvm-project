//===-- Unittests for __sched_cpualloc and __sched_cpufree ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sched_macros.h"
#include "hdr/types/cpu_set_t.h"
#include "src/sched/sched_cpualloc.h"
#include "src/sched/sched_cpufree.h"
#include "src/sched/sched_getcpucount.h"
#include "src/sched/sched_getcpuisset.h"
#include "src/sched/sched_setcpuset.h"
#include "src/sched/sched_setcpuzero.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

using LlvmLibcSchedCpuAllocTest = testing::Test;

TEST_F(LlvmLibcSchedCpuAllocTest, AllocAndFreeSmall) {
  int num_cpus = 10;
  size_t alloc_size = CPU_ALLOC_SIZE(num_cpus);
  cpu_set_t *mask = CPU_ALLOC(num_cpus);
  ASSERT_NE(mask, static_cast<cpu_set_t *>(nullptr));

  CPU_ZERO_S(alloc_size, mask);
  ASSERT_EQ(CPU_COUNT_S(alloc_size, mask), 0);
  ASSERT_EQ(CPU_ISSET_S(1, alloc_size, mask), 0);

  CPU_SET_S(1, alloc_size, mask);
  ASSERT_EQ(CPU_ISSET_S(1, alloc_size, mask), 1);
  ASSERT_EQ(CPU_COUNT_S(alloc_size, mask), 1);

  CPU_SET_S(5, alloc_size, mask);
  ASSERT_EQ(CPU_ISSET_S(5, alloc_size, mask), 1);
  ASSERT_EQ(CPU_COUNT_S(alloc_size, mask), 2);

  CPU_FREE(mask);
}

TEST_F(LlvmLibcSchedCpuAllocTest, AllocAndFreeLarge) {
  int num_cpus = 4096;
  size_t alloc_size = CPU_ALLOC_SIZE(num_cpus);
  cpu_set_t *mask = CPU_ALLOC(num_cpus);
  ASSERT_NE(mask, static_cast<cpu_set_t *>(nullptr));

  CPU_ZERO_S(alloc_size, mask);
  ASSERT_EQ(CPU_COUNT_S(alloc_size, mask), 0);
  ASSERT_EQ(CPU_ISSET_S(1, alloc_size, mask), 0);

  CPU_SET_S(1, alloc_size, mask);
  ASSERT_EQ(CPU_ISSET_S(1, alloc_size, mask), 1);
  ASSERT_EQ(CPU_COUNT_S(alloc_size, mask), 1);

  CPU_SET_S(500, alloc_size, mask);
  ASSERT_EQ(CPU_ISSET_S(500, alloc_size, mask), 1);
  ASSERT_EQ(CPU_COUNT_S(alloc_size, mask), 2);

  CPU_SET_S(4095, alloc_size, mask);
  ASSERT_EQ(CPU_ISSET_S(4095, alloc_size, mask), 1);
  ASSERT_EQ(CPU_COUNT_S(alloc_size, mask), 3);

  CPU_FREE(mask);
}

} // namespace LIBC_NAMESPACE_DECL
