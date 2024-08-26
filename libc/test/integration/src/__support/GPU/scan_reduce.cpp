//===-- Test for the parallel scan and reduction operations on the GPU ----===//
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

static uint32_t sum(uint32_t n) { return n * (n + 1) / 2; }

// Tests a reduction within a convergant warp or wavefront using some known
// values. For example, if every element in the lane is one, then the sum should
// be the size of the warp or wavefront, i.e. 1 + 1 + 1 ... + 1.
static void test_reduce() {
  uint64_t mask = gpu::get_lane_mask();
  uint32_t x = gpu::reduce(mask, 1);
  EXPECT_EQ(x, gpu::get_lane_size());

  uint32_t y = gpu::reduce(mask, gpu::get_lane_id());
  EXPECT_EQ(y, sum(gpu::get_lane_size() - 1));

  uint32_t z = 0;
  if (gpu::get_lane_id() % 2)
    z = gpu::reduce(gpu::get_lane_mask(), 1);
  gpu::sync_lane(mask);

  EXPECT_EQ(z, gpu::get_lane_id() % 2 ? gpu::get_lane_size() / 2 : 0);
}

// Tests an accumulation scan within a convergent warp or wavefront using some
// known values. For example, if every element in the lane is one, then the scan
// should have each element be equivalent to its ID, i.e. 1, 1 + 1, ...
static void test_scan() {
  uint64_t mask = gpu::get_lane_mask();

  uint32_t x = gpu::scan(mask, 1);
  EXPECT_EQ(x, gpu::get_lane_id() + 1);

  uint32_t y = gpu::scan(mask, gpu::get_lane_id());
  EXPECT_EQ(y, sum(gpu::get_lane_id()));

  uint32_t z = 0;
  if (gpu::get_lane_id() % 2)
    z = gpu::scan(gpu::get_lane_mask(), 1);
  gpu::sync_lane(mask);

  EXPECT_EQ(z, gpu::get_lane_id() % 2 ? gpu::get_lane_id() / 2 + 1 : 0);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  test_reduce();

  test_scan();

  return 0;
}
