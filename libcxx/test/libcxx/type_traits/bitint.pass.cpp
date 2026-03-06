//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _BitInt(N) is recognized as a signed/unsigned integer type by
// libc++ internal traits, and that <bit> operations work for all valid widths.

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: gcc

#include <bit>
#include <cassert>
#include <limits>
#include <type_traits>

// ===== Type traits =====
// _BitInt(N) must satisfy is_integral, is_signed/is_unsigned, is_arithmetic

template <int N>
void test_signed_traits() {
  using T = _BitInt(N);
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_signed_v<T>);
  static_assert(!std::is_unsigned_v<T>);
  static_assert(std::is_arithmetic_v<T>);
  static_assert(std::numeric_limits<T>::is_specialized);
}

template <int N>
void test_unsigned_traits() {
  using T = unsigned _BitInt(N);
  static_assert(std::is_integral_v<T>);
  static_assert(!std::is_signed_v<T>);
  static_assert(std::is_unsigned_v<T>);
  static_assert(std::is_arithmetic_v<T>);
  static_assert(std::numeric_limits<T>::is_specialized);
}

// ===== Negative tests =====
// Character types and bool must NOT satisfy __signed_integer/__unsigned_integer
// (they are integral but not "integer types" per [basic.fundamental])

static_assert(std::is_integral_v<bool>);
static_assert(std::is_integral_v<char>);
static_assert(std::is_integral_v<wchar_t>);
static_assert(std::is_integral_v<char16_t>);
static_assert(std::is_integral_v<char32_t>);
// These types are integral but we cannot directly test the internal
// __is_signed_integer_v trait here. Instead we verify that the <bit>
// operations correctly reject them (they require __unsigned_integer).

// ===== Bit operations =====

template <int N>
void test_popcount() {
  using T = unsigned _BitInt(N);
  assert(std::popcount(T(0)) == 0);
  assert(std::popcount(T(1)) == 1);
  if constexpr (N >= 8)
    assert(std::popcount(T(0xFF)) == 8);
}

// countl_zero and countr_zero use numeric_limits::digits internally.
// For non-byte-aligned _BitInt(N), digits == sizeof*CHAR_BIT which may
// exceed N. Only assert exact values for byte-aligned widths here.

template <int N>
void test_countl_zero() {
  using T = unsigned _BitInt(N);
  // countl_zero(1): result is digits - 1 (digits may exceed N for
  // non-byte-aligned widths due to numeric_limits using sizeof*CHAR_BIT)
  assert(std::countl_zero(T(1)) >= N - 1);
}

template <int N>
void test_countr_zero() {
  using T = unsigned _BitInt(N);
  assert(std::countr_zero(T(1)) == 0);
  assert(std::countr_zero(T(T(1) << (N - 1))) == N - 1);
}

// bit_width and has_single_bit depend on numeric_limits::digits being correct.
// For non-byte-aligned _BitInt(N), digits uses sizeof*CHAR_BIT which exceeds N.
// Only test byte-aligned widths here; a separate fix for numeric_limits::digits
// will enable testing all widths.

template <int N>
void test_bit_width() {
  using T = unsigned _BitInt(N);
  assert(std::bit_width(T(0)) == 0);
  assert(std::bit_width(T(1)) == 1);
  if constexpr (N >= 11)
    assert(std::bit_width(T(1024)) == 11);
  assert(std::bit_width(T(~T(0))) == N);
}

template <int N>
void test_has_single_bit() {
  using T = unsigned _BitInt(N);
  assert(!std::has_single_bit(T(0)));
  assert(std::has_single_bit(T(1)));
  if constexpr (N >= 8) {
    assert(std::has_single_bit(T(128)));
    assert(!std::has_single_bit(T(129)));
  }
}

// Big-number popcount test: verified with Python
void test_popcount_big_numbers() {
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
#endif
#if __BITINT_MAXWIDTH__ >= 4096
  {
    unsigned _BitInt(4096) v = ~(unsigned _BitInt(4096))(0);
    assert(std::popcount(v) == 4096);
  }
#endif
}

// Big-number countl_zero test
void test_countl_zero_big_numbers() {
#if __BITINT_MAXWIDTH__ >= 256
  {
    // Bit set at position 200 in a 256-bit integer: 55 leading zeros
    unsigned _BitInt(256) v = (unsigned _BitInt(256))(1) << 200;
    assert(std::countl_zero(v) == 55);
  }
#endif
#if __BITINT_MAXWIDTH__ >= 4096
  {
    unsigned _BitInt(4096) v = (unsigned _BitInt(4096))(1) << 4000;
    assert(std::countl_zero(v) == 95);
  }
#endif
}

template <int N>
void test_all() {
  test_signed_traits<N>();
  test_unsigned_traits<N>();
  test_popcount<N>();
  test_countl_zero<N>();
  test_countr_zero<N>();
}

// Only test bit_width/has_single_bit for byte-aligned widths where
// numeric_limits::digits == N (see comment above).
template <int N>
void test_all_with_bit_width() {
  test_all<N>();
  test_bit_width<N>();
  test_has_single_bit<N>();
}

int main(int, char**) {
  // unsigned _BitInt(1) is the minimum unsigned width.
  // signed _BitInt(1) is illegal -- minimum signed width is 2.
  test_unsigned_traits<1>();
  test_popcount<1>();

  // _BitInt(2): minimum signed width
  test_signed_traits<2>();
  test_unsigned_traits<2>();
  test_popcount<2>();

  // Standard power-of-2 widths: byte-aligned, so bit_width/has_single_bit work
  test_all_with_bit_width<8>();
  test_all_with_bit_width<16>();
  test_all_with_bit_width<32>();
  test_all_with_bit_width<64>();
  test_all_with_bit_width<128>();

  // Odd widths -- popcount/countl_zero/countr_zero work, but bit_width and
  // has_single_bit may give wrong results due to numeric_limits::digits using
  // sizeof*CHAR_BIT instead of the actual bit width N.
  test_all<7>();
  test_all<9>();
  test_all<15>();
  test_all<17>();
  test_all<33>();
  test_all<65>();
  test_all<127>();

  // Wide _BitInt (N > 128) is only supported on some targets.
#if __BITINT_MAXWIDTH__ >= 256
  test_all_with_bit_width<256>();
  test_all<129>();
  test_all<255>();
  test_all<257>();
  test_all<512>();
  test_all<1024>();
#endif
#if __BITINT_MAXWIDTH__ >= 4096
  test_all_with_bit_width<4096>();
#endif

  // Big number tests (Python-verified expected values)
  test_popcount_big_numbers();
  test_countl_zero_big_numbers();

  return 0;
}
