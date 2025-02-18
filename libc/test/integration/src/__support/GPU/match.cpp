//===-- Test for the shuffle operations on the GPU ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h"
#include "src/__support/GPU/utils.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

// Test to ensure that match any / match all work.
static void test_match() {
  uint64_t mask = gpu::get_lane_mask();
  EXPECT_EQ(1ull << gpu::get_lane_id(),
            gpu::match_any(mask, gpu::get_lane_id()));
  EXPECT_EQ(mask, gpu::match_any(mask, 1));

  uint64_t expected = gpu::get_lane_id() < 16 ? 0xffff : 0xffff0000;
  EXPECT_EQ(expected, gpu::match_any(mask, gpu::get_lane_id() < 16));
  EXPECT_EQ(mask, gpu::match_all(mask, 1));
  EXPECT_EQ(0ull, gpu::match_all(mask, gpu::get_lane_id()));
}

TEST_MAIN(int argc, char **argv, char **envp) {
  if (gpu::get_thread_id() >= gpu::get_lane_size())
    return 0;

  test_match();

  return 0;
}
