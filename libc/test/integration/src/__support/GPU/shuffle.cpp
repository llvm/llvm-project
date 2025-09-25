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

// Test to make sure the shuffle instruction works by doing a simple broadcast.
// Each iteration reduces the width, so it will broadcast to a subset we check.
static void test_shuffle() {
  uint64_t mask = gpu::get_lane_mask();
  EXPECT_EQ(cpp::popcount(mask), gpu::get_lane_size());

  uint32_t x = gpu::get_lane_id();
  for (uint32_t width = gpu::get_lane_size(); width > 0; width /= 2)
    EXPECT_EQ(gpu::shuffle(mask, 0, x, width), (x / width) * width);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  if (gpu::get_thread_id() >= gpu::get_lane_size())
    return 0;

  test_shuffle();

  return 0;
}
