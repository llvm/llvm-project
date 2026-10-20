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

static void test_reduce_bitwise() {
  uint64_t mask = gpu::get_lane_mask();
  uint32_t id = gpu::get_lane_id();

  EXPECT_EQ(__gpu_lane_and_u32(mask, 0xFFu), 0xFFu);
  EXPECT_EQ(__gpu_lane_and_u32(mask, id == 0 ? 0x0Fu : 0xFFu), 0x0Fu);

  EXPECT_EQ(__gpu_lane_or_u32(mask, id == 0 ? 0xF0u : 0x0Fu), 0xFFu);
  EXPECT_EQ(__gpu_lane_or_u32(mask, 0u), 0u);

  EXPECT_EQ(__gpu_lane_xor_u32(mask, 1u), 0u);
  EXPECT_EQ(__gpu_lane_xor_u32(mask, id == 0 ? 0xFFu : 0u), 0xFFu);
}

static void test_reduce_min_max() {
  uint64_t mask = gpu::get_lane_mask();
  uint32_t id = gpu::get_lane_id();
  uint32_t n = gpu::get_lane_size();

  EXPECT_EQ(__gpu_lane_min_u32(mask, id), 0u);
  EXPECT_EQ(__gpu_lane_max_u32(mask, id), n - 1);
  EXPECT_EQ(__gpu_lane_min_u32(mask, n - 1 - id), 0u);
  EXPECT_EQ(__gpu_lane_max_u32(mask, n - 1 - id), n - 1);

  EXPECT_EQ(__gpu_lane_minnum_f32(mask, static_cast<float>(id)), 0.0f);
  EXPECT_EQ(__gpu_lane_maxnum_f32(mask, static_cast<float>(id)),
            static_cast<float>(n - 1));
}

static void test_scan_bitwise() {
  uint64_t mask = gpu::get_lane_mask();
  uint32_t id = gpu::get_lane_id();

  EXPECT_EQ(__gpu_prefix_scan_and_u32(mask, 0xFFu), 0xFFu);
  EXPECT_EQ(__gpu_prefix_scan_and_u32(mask, id == 0 ? 0x0Fu : 0xFFu), 0x0Fu);

  EXPECT_EQ(__gpu_prefix_scan_or_u32(mask, 0x0Fu), 0x0Fu);
  uint32_t or_expected = id == 0 ? 0xF0u : 0xFFu;
  EXPECT_EQ(__gpu_prefix_scan_or_u32(mask, id == 0 ? 0xF0u : 0x0Fu),
            or_expected);

  uint32_t xor_expected = id % 2 == 0 ? 0x0Fu : 0u;
  EXPECT_EQ(__gpu_prefix_scan_xor_u32(mask, 0x0Fu), xor_expected);
}

static void test_scan_min_max() {
  uint64_t mask = gpu::get_lane_mask();
  uint32_t id = gpu::get_lane_id();
  uint32_t n = gpu::get_lane_size();

  EXPECT_EQ(__gpu_prefix_scan_min_u32(mask, n - 1 - id), n - 1 - id);
  EXPECT_EQ(__gpu_prefix_scan_max_u32(mask, id), id);

  EXPECT_EQ(__gpu_prefix_scan_min_u32(mask, id), 0u);
  EXPECT_EQ(__gpu_prefix_scan_max_u32(mask, n - 1 - id), n - 1);

  EXPECT_EQ(__gpu_prefix_scan_minnum_f32(mask, static_cast<float>(n - 1 - id)),
            static_cast<float>(n - 1 - id));
  EXPECT_EQ(__gpu_prefix_scan_maxnum_f32(mask, static_cast<float>(id)),
            static_cast<float>(id));
}

static void test_float_min_max() {
  uint64_t mask = gpu::get_lane_mask();
  uint32_t id = gpu::get_lane_id();
  uint32_t n = gpu::get_lane_size();

  float centered = static_cast<float>(id) - static_cast<float>(n / 2);
  EXPECT_EQ(__gpu_lane_minnum_f32(mask, centered), -static_cast<float>(n / 2));
  EXPECT_EQ(__gpu_lane_maxnum_f32(mask, centered),
            static_cast<float>(n / 2 - 1));

  float alt =
      id % 2 == 0 ? static_cast<float>(id + 1) : -static_cast<float>(id + 1);
  EXPECT_EQ(__gpu_lane_minnum_f32(mask, alt), -static_cast<float>(n));
  EXPECT_EQ(__gpu_lane_maxnum_f32(mask, alt), static_cast<float>(n - 1));

  float v_val = id < n / 2 ? static_cast<float>(n / 2 - id)
                           : static_cast<float>(id - n / 2);
  float min_expected = id < n / 2 ? static_cast<float>(n / 2 - id) : 0.0f;
  EXPECT_EQ(__gpu_prefix_scan_minnum_f32(mask, v_val), min_expected);

  float inv_v =
      id < n / 2 ? static_cast<float>(id) : static_cast<float>(n - 1 - id);
  float max_expected =
      id < n / 2 ? static_cast<float>(id) : static_cast<float>(n / 2 - 1);
  EXPECT_EQ(__gpu_prefix_scan_maxnum_f32(mask, inv_v), max_expected);

  double d_centered = static_cast<double>(id) - static_cast<double>(n / 2);
  EXPECT_EQ(__gpu_lane_minnum_f64(mask, d_centered),
            -static_cast<double>(n / 2));
  EXPECT_EQ(__gpu_lane_maxnum_f64(mask, d_centered),
            static_cast<double>(n / 2 - 1));

  double desc = static_cast<double>(n - 1 - id);
  EXPECT_EQ(__gpu_prefix_scan_minnum_f64(mask, desc),
            static_cast<double>(n - 1 - id));
  EXPECT_EQ(__gpu_prefix_scan_maxnum_f64(mask, desc),
            static_cast<double>(n - 1));
}

TEST_MAIN(int, char **, char **) {
  if (gpu::get_thread_id() >= gpu::get_lane_size())
    return 0;

  test_reduce();
  test_reduce_bitwise();
  test_reduce_min_max();

  test_scan();
  test_scan_bitwise();
  test_scan_min_max();

  test_float_min_max();

  test_scan_divergent();

  return 0;
}
