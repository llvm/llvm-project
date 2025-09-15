//===-- Test for parallel GPU malloc interface ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"

#include "src/__support/GPU/utils.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"

using namespace LIBC_NAMESPACE;

TEST_MAIN(int, char **, char **) {
  int *convergent = reinterpret_cast<int *>(LIBC_NAMESPACE::malloc(16));
  EXPECT_NE(convergent, nullptr);
  *convergent = 1;
  EXPECT_EQ(*convergent, 1);
  LIBC_NAMESPACE::free(convergent);

  int *divergent = reinterpret_cast<int *>(
      LIBC_NAMESPACE::malloc((gpu::get_thread_id() + 1) * 16));
  EXPECT_NE(divergent, nullptr);
  EXPECT_TRUE(__builtin_is_aligned(divergent, 16));
  *divergent = 1;
  EXPECT_EQ(*divergent, 1);
  LIBC_NAMESPACE::free(divergent);

  if (gpu::get_lane_id() & 1) {
    int *masked = reinterpret_cast<int *>(
        LIBC_NAMESPACE::malloc((gpu::get_thread_id() + 1) * 16));
    EXPECT_NE(masked, nullptr);
    *masked = 1;
    EXPECT_EQ(*masked, 1);
    LIBC_NAMESPACE::free(masked);
  }
  return 0;
}
