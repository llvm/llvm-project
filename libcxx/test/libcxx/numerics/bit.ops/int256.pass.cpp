//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Test <bit> operations with __uint256_t (Tier 2 -- key for Hamming distance)

#include <bit>
#include <limits>
#include <type_traits>

#include "test_macros.h"

#ifdef TEST_HAS_NO_INT256
int main(int, char**) { return 0; }
#else

// std::popcount -- the core operation for Hamming distance in neural search
static_assert(std::popcount((__uint256_t)0) == 0);
static_assert(std::popcount((__uint256_t)1) == 1);
static_assert(std::popcount((__uint256_t)0xFF) == 8);
static_assert(std::popcount((__uint256_t)0xFFFFFFFFFFFFFFFF) == 64);

// std::countl_zero
static_assert(std::countl_zero((__uint256_t)0) == 256);
static_assert(std::countl_zero((__uint256_t)1) == 255);

// std::countr_zero
static_assert(std::countr_zero((__uint256_t)0) == 256);
static_assert(std::countr_zero((__uint256_t)1) == 0);
static_assert(std::countr_zero((__uint256_t)2) == 1);

// std::countl_one
static_assert(std::countl_one((__uint256_t)0) == 0);

// std::countr_one
static_assert(std::countr_one((__uint256_t)0) == 0);
static_assert(std::countr_one((__uint256_t)1) == 1);
static_assert(std::countr_one((__uint256_t)0xFF) == 8);

// std::has_single_bit
static_assert(std::has_single_bit((__uint256_t)1));
static_assert(std::has_single_bit((__uint256_t)2));
static_assert(std::has_single_bit((__uint256_t)4));
static_assert(!std::has_single_bit((__uint256_t)3));
static_assert(!std::has_single_bit((__uint256_t)0));

// std::bit_width
static_assert(std::bit_width((__uint256_t)0) == 0);
static_assert(std::bit_width((__uint256_t)1) == 1);
static_assert(std::bit_width((__uint256_t)2) == 2);
static_assert(std::bit_width((__uint256_t)255) == 8);

// std::rotl / std::rotr
static_assert(std::rotl((__uint256_t)1, 1) == 2);
static_assert(std::rotl((__uint256_t)1, 64) == ((__uint256_t)1 << 64));
static_assert(std::rotl((__uint256_t)1, 255) == ((__uint256_t)1 << 255));
static_assert(std::rotr((__uint256_t)2, 1) == 1);
static_assert(std::rotr((__uint256_t)1, 1) == ((__uint256_t)1 << 255));
static_assert(std::rotl(std::rotr((__uint256_t)0xFF, 4), 4) == 0xFF);

// std::bit_ceil
static_assert(std::bit_ceil((__uint256_t)0) == 1);
static_assert(std::bit_ceil((__uint256_t)1) == 1);
static_assert(std::bit_ceil((__uint256_t)2) == 2);
static_assert(std::bit_ceil((__uint256_t)3) == 4);
static_assert(std::bit_ceil((__uint256_t)255) == 256);

// std::bit_floor
static_assert(std::bit_floor((__uint256_t)0) == 0);
static_assert(std::bit_floor((__uint256_t)1) == 1);
static_assert(std::bit_floor((__uint256_t)2) == 2);
static_assert(std::bit_floor((__uint256_t)3) == 2);
static_assert(std::bit_floor((__uint256_t)255) == 128);

int main(int, char**) {
  // Runtime: Hamming distance pattern (Algolia neural search style)
  __uint256_t a = (__uint256_t)0xDEADBEEF << 128 | 0xCAFEBABE;
  __uint256_t b = (__uint256_t)0xFEEDFACE << 128 | 0xBAADF00D;
  int hamming   = std::popcount(a ^ b);
  (void)hamming;

  // Runtime: Verify popcount of known pattern
  __uint256_t all_ones_low64 = 0xFFFFFFFFFFFFFFFF;
  if (std::popcount(all_ones_low64) != 64)
    return 1;

  __uint256_t all_zeros = 0;
  if (std::popcount(all_zeros) != 0)
    return 2;

  return 0;
}
#endif
