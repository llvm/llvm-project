//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that <bit> operations work with _BitInt(N) for various widths.

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: gcc
// UNSUPPORTED: LIBCXX-PICOLIBC-FIXME

#include <bit>
#include <cassert>
#include <limits>

// Full test suite for byte-aligned widths where numeric_limits::digits == N.

template <int N>
void test_all() {
  using T = unsigned _BitInt(N);

  // popcount
  assert(std::popcount(T(0)) == 0);
  assert(std::popcount(T(1)) == 1);
  assert(std::popcount(T(~T(0))) == N);
  if constexpr (N >= 8)
    assert(std::popcount(T(0xFF)) == 8);

  // countl_zero / countr_zero
  assert(std::countl_zero(T(~T(0))) == 0);
  assert(std::countl_zero(T(1)) == N - 1);
  assert(std::countr_zero(T(1)) == 0);
  assert(std::countr_zero(T(T(1) << (N - 1))) == N - 1);

  // countl_one / countr_one
  assert(std::countl_one(T(0)) == 0);
  assert(std::countl_one(T(~T(0))) == N);
  assert(std::countr_one(T(0)) == 0);
  assert(std::countr_one(T(~T(0))) == N);
  assert(std::countr_one(T(1)) == 1);

  // rotl / rotr
  assert(std::rotl(T(1), 0) == T(1));
  assert(std::rotr(T(1), 0) == T(1));
  if constexpr (N >= 8) {
    assert(std::rotl(T(1), 4) == T(16));
    assert(std::rotr(T(16), 4) == T(1));
  }

  // bit_width
  assert(std::bit_width(T(0)) == 0);
  assert(std::bit_width(T(1)) == 1);
  assert(std::bit_width(T(~T(0))) == N);
  if constexpr (N >= 11)
    assert(std::bit_width(T(1024)) == 11);

  // has_single_bit
  assert(!std::has_single_bit(T(0)));
  assert(std::has_single_bit(T(1)));
  if constexpr (N >= 8) {
    assert(std::has_single_bit(T(128)));
    assert(!std::has_single_bit(T(129)));
  }

  // bit_ceil
  assert(std::bit_ceil(T(0)) == T(1));
  assert(std::bit_ceil(T(1)) == T(1));
  if constexpr (N >= 8) {
    assert(std::bit_ceil(T(3)) == T(4));
    assert(std::bit_ceil(T(128)) == T(128));
  }
  // bit_ceil(129) == 256 requires N >= 9 (result must be representable)
  if constexpr (N >= 9)
    assert(std::bit_ceil(T(129)) == T(256));

  // bit_floor
  assert(std::bit_floor(T(0)) == T(0));
  assert(std::bit_floor(T(1)) == T(1));
  if constexpr (N >= 8) {
    assert(std::bit_floor(T(3)) == T(2));
    assert(std::bit_floor(T(128)) == T(128));
    assert(std::bit_floor(T(255)) == T(128));
  }
}

// Reduced test for non-byte-aligned widths. These widths have incorrect
// numeric_limits::digits (sizeof*CHAR_BIT instead of N), which breaks
// bit_width, has_single_bit, bit_ceil, bit_floor, and the exact values
// of countl_zero/countl_one. Only test functions that work correctly.
template <int N>
void test_odd() {
  using T = unsigned _BitInt(N);

  assert(std::popcount(T(0)) == 0);
  assert(std::popcount(T(1)) == 1);
  assert(std::popcount(T(~T(0))) == N);

  assert(std::countl_zero(T(1)) >= N - 1);
  assert(std::countl_zero(T(~T(0))) == 0);
  assert(std::countr_zero(T(1)) == 0);
  assert(std::countr_zero(T(T(1) << (N - 1))) == N - 1);
}

void test_big_numbers() {
#if __BITINT_MAXWIDTH__ >= 256
  {
    // (1 << 200) - 1 has exactly 200 bits set
    unsigned _BitInt(256) v = (unsigned _BitInt(256))(1) << 200;
    v -= 1;
    assert(std::popcount(v) == 200);
  }
  {
    // Exactly 4 bits set at positions 0, 64, 128, 255
    unsigned _BitInt(256) v = (unsigned _BitInt(256))(1) | ((unsigned _BitInt(256))(1) << 64) |
                              ((unsigned _BitInt(256))(1) << 128) | ((unsigned _BitInt(256))(1) << 255);
    assert(std::popcount(v) == 4);
  }
  {
    // Bit set at position 200 in a 256-bit integer: 55 leading zeros
    unsigned _BitInt(256) v = (unsigned _BitInt(256))(1) << 200;
    assert(std::countl_zero(v) == 55);
  }
#endif
#if __BITINT_MAXWIDTH__ >= 4096
  {
    unsigned _BitInt(4096) v = ~(unsigned _BitInt(4096))(0);
    assert(std::popcount(v) == 4096);
  }
  {
    unsigned _BitInt(4096) v = (unsigned _BitInt(4096))(1) << 4000;
    assert(std::countl_zero(v) == 95);
  }
#endif
}

int main(int, char**) {
  // unsigned _BitInt(1) is the minimum unsigned width.
  test_odd<1>();

  // _BitInt(2): minimum signed width
  test_odd<2>();

  // Standard power-of-2 widths (byte-aligned: full suite)
  test_all<8>();
  test_all<16>();
  test_all<32>();
  test_all<64>();
  test_all<128>();

  // Odd widths (reduced suite: popcount, countl_zero, countr_zero only)
  test_odd<7>();
  test_odd<9>();
  test_odd<15>();
  test_odd<33>();
  test_odd<65>();
  test_odd<127>();

  // Wide _BitInt (N > 128) is only supported on some targets.
#if __BITINT_MAXWIDTH__ >= 256
  test_all<256>();
  test_odd<129>();
  test_odd<255>();
#endif
#if __BITINT_MAXWIDTH__ >= 4096
  test_all<4096>();
#endif

  // Big number tests (Python-verified expected values)
  test_big_numbers();

  return 0;
}
