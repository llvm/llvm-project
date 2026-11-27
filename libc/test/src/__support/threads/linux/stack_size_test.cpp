//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/limits_macros.h"
#include "hdr/pthread_macros.h"
#include "hdr/sys_resource_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_rlimit.h"
#include "src/__support/threads/stack_size.h"
#include "test/UnitTest/Test.h"

#include <linux/param.h>

namespace {

using LIBC_NAMESPACE::detail::FALLBACK_STACKSIZE;
using LIBC_NAMESPACE::detail::stacksize_from_rlimit;

constexpr size_t PAGE_SIZE = static_cast<size_t>(EXEC_PAGESIZE);
constexpr size_t PAGE_MASK = PAGE_SIZE - 1;
constexpr size_t MAX_STACKSIZE = ~PAGE_MASK - PAGE_SIZE;
constexpr size_t MIN_STACKSIZE = static_cast<size_t>(PTHREAD_STACK_MIN);
constexpr size_t ALIGNED_MIN_STACKSIZE =
    (MIN_STACKSIZE + PAGE_MASK) & ~PAGE_MASK;

TEST(LlvmLibcSupportThreadsStackSizeTest, BelowAndAtMinimum) {
  EXPECT_EQ(stacksize_from_rlimit(0), ALIGNED_MIN_STACKSIZE);
  EXPECT_EQ(stacksize_from_rlimit(1), ALIGNED_MIN_STACKSIZE);
  EXPECT_EQ(stacksize_from_rlimit(MIN_STACKSIZE - 1), ALIGNED_MIN_STACKSIZE);
  EXPECT_EQ(stacksize_from_rlimit(MIN_STACKSIZE), ALIGNED_MIN_STACKSIZE);
}

TEST(LlvmLibcSupportThreadsStackSizeTest, PageAlignment) {
  EXPECT_EQ(stacksize_from_rlimit(ALIGNED_MIN_STACKSIZE + 1),
            ALIGNED_MIN_STACKSIZE + PAGE_SIZE);
  EXPECT_EQ(stacksize_from_rlimit(ALIGNED_MIN_STACKSIZE + PAGE_MASK),
            ALIGNED_MIN_STACKSIZE + PAGE_SIZE);
  EXPECT_EQ(stacksize_from_rlimit(ALIGNED_MIN_STACKSIZE + PAGE_SIZE),
            ALIGNED_MIN_STACKSIZE + PAGE_SIZE);
}

TEST(LlvmLibcSupportThreadsStackSizeTest, MaxStacksizeBoundaries) {
  EXPECT_EQ(stacksize_from_rlimit(MAX_STACKSIZE - PAGE_SIZE),
            MAX_STACKSIZE - PAGE_SIZE);
  EXPECT_EQ(stacksize_from_rlimit(MAX_STACKSIZE - PAGE_MASK), MAX_STACKSIZE);
  EXPECT_EQ(stacksize_from_rlimit(MAX_STACKSIZE - 1), MAX_STACKSIZE);
  EXPECT_EQ(stacksize_from_rlimit(MAX_STACKSIZE), MAX_STACKSIZE);
}

TEST(LlvmLibcSupportThreadsStackSizeTest, AboveMaxStacksizeFallback) {
  EXPECT_EQ(stacksize_from_rlimit(static_cast<rlim_t>(MAX_STACKSIZE) + 1),
            FALLBACK_STACKSIZE);
  EXPECT_EQ(stacksize_from_rlimit(RLIM_INFINITY - 1), FALLBACK_STACKSIZE);
}

TEST(LlvmLibcSupportThreadsStackSizeTest, RlimInfinity) {
  EXPECT_EQ(stacksize_from_rlimit(RLIM_INFINITY), FALLBACK_STACKSIZE);
}

} // namespace
