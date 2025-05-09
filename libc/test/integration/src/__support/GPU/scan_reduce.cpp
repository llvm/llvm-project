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

static uint32_t random(uint64_t *rand_next) {
  uint64_t x = *rand_next;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *rand_next = x;
  return static_cast<uint32_t>((x * 0x2545F4914F6CDD1Dul) >> 32);
}

// Scan operations can break down under thread divergence, make sure that the
// function works under some random divergence. We do this by trivially
// implementing a scan with shared scratch memory and then comparing the
// results.
static void test_scan_divergent() {
  static uint32_t input[64] = {0};
  static uint32_t result[64] = {0};
  uint64_t state = gpu::processor_clock() + __gpu_lane_id();

  for (int i = 0; i < 64; ++i) {
    uint64_t lanemask = gpu::get_lane_mask();
    if (random(&state) & (1ull << gpu::get_lane_id())) {
      uint64_t divergent = gpu::get_lane_mask();
      uint32_t value = random(&state) % 256;
      input[gpu::get_lane_id()] = value;

      if (gpu::is_first_lane(divergent)) {
        uint32_t accumulator = 0;
        for (uint32_t lane = 0; lane < gpu::get_lane_size(); ++lane) {
          uint32_t tmp = input[lane];
          result[lane] = tmp + accumulator;
          accumulator += tmp;
        }
      }
      gpu::sync_lane(divergent);

      uint32_t scan = gpu::scan(divergent, value);
      EXPECT_EQ(scan, result[gpu::get_lane_id()]);
    }
    if (gpu::is_first_lane(lanemask))
      __builtin_memset(input, 0, sizeof(input));
    gpu::sync_lane(lanemask);
  }
}

TEST_MAIN(int argc, char **argv, char **envp) {
  if (gpu::get_thread_id() >= gpu::get_lane_size())
    return 0;

  test_reduce();

  test_scan();

  test_scan_divergent();

  return 0;
}
