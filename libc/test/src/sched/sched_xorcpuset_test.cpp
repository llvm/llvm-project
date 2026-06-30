//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for __sched_xorcpuset.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/cpu_set_t.h"
#include "src/sched/sched_getcpucount.h"
#include "src/sched/sched_getcpuisset.h"
#include "src/sched/sched_setcpuset.h"
#include "src/sched/sched_setcpuzero.h"
#include "src/sched/sched_xorcpuset.h"
#include "test/UnitTest/Test.h"

using LlvmLibcSchedXorCpuSetTest = LIBC_NAMESPACE::testing::Test;

TEST_F(LlvmLibcSchedXorCpuSetTest, XorCpuSet) {
  cpu_set_t mask1, mask2, result;
  const size_t alloc_size = sizeof(cpu_set_t);

  LIBC_NAMESPACE::__sched_setcpuzero(alloc_size, &mask1);
  LIBC_NAMESPACE::__sched_setcpuzero(alloc_size, &mask2);
  LIBC_NAMESPACE::__sched_setcpuset(1, alloc_size, &mask1);
  LIBC_NAMESPACE::__sched_setcpuset(1, alloc_size, &mask2);
  LIBC_NAMESPACE::__sched_setcpuset(2, alloc_size, &mask2);

  LIBC_NAMESPACE::__sched_xorcpuset(alloc_size, &result, &mask1, &mask2);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpuisset(1, alloc_size, &result), 0);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpuisset(2, alloc_size, &result), 1);
  ASSERT_EQ(LIBC_NAMESPACE::__sched_getcpucount(alloc_size, &result), 1);
}
