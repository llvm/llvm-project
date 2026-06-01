//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "test_macros.h"

template <class T>
concept has_byteswap = requires(T t) {
  std::byteswap(t);
};

static_assert(!has_byteswap<void*>);
static_assert(!has_byteswap<float>);
static_assert(!has_byteswap<char[2]>);
static_assert(!has_byteswap<std::byte>);

// _BitInt(N) candidacy is controlled by the `integral` constraint; the
// padding-bit rejection (per [bit.byteswap]/Mandates) is a static_assert
// inside the function body, not a SFINAE check. See byteswap.verify.cpp
// for diagnostic verification of the rejected widths.

template <class T>
constexpr void test_num(T in, T expected) {
  assert(std::byteswap(in) == expected);
  ASSERT_SAME_TYPE(decltype(std::byteswap(in)), decltype(in));
  ASSERT_NOEXCEPT(std::byteswap(in));
}

template <class T>
constexpr std::pair<T, T> get_test_data() {
  switch (sizeof(T)) {
  case 2:
    return {static_cast<T>(0x1234), static_cast<T>(0x3412)};
  case 4:
    return {static_cast<T>(0x60AF8503), static_cast<T>(0x0385AF60)};
  case 8:
    return {static_cast<T>(0xABCDFE9477936406), static_cast<T>(0x0664937794FECDAB)};
  default:
    assert(false);
    return {}; // for MSVC, whose `assert` is tragically not [[noreturn]]
  }
}

template <class T>
constexpr void test_implementation_defined_size() {
  const auto [in, expected] = get_test_data<T>();
  test_num<T>(in, expected);
}

constexpr bool test() {
  test_num<std::uint8_t>(0xAB, 0xAB);
  test_num<std::uint16_t>(0xCDEF, 0xEFCD);
  test_num<std::uint32_t>(0x01234567, 0x67452301);
  test_num<std::uint64_t>(0x0123456789ABCDEF, 0xEFCDAB8967452301);

  test_num<std::int8_t>(static_cast<std::int8_t>(0xAB), static_cast<std::int8_t>(0xAB));
  test_num<std::int16_t>(static_cast<std::int16_t>(0xCDEF), static_cast<std::int16_t>(0xEFCD));
  test_num<std::int32_t>(0x01234567, 0x67452301);
  test_num<std::int64_t>(0x0123456789ABCDEF, 0xEFCDAB8967452301);

#ifndef TEST_HAS_NO_INT128
  const auto in = static_cast<__uint128_t>(0x0123456789ABCDEF) << 64 | 0x13579BDF02468ACE;
  const auto expected = static_cast<__uint128_t>(0xCE8A4602DF9B5713) << 64 | 0xEFCDAB8967452301;
  test_num<__uint128_t>(in, expected);
  test_num<__int128_t>(in, expected);
#endif

  test_num<bool>(true, true);
  test_num<bool>(false, false);
  test_num<char>(static_cast<char>(0xCD), static_cast<char>(0xCD));
  test_num<unsigned char>(0xEF, 0xEF);
  test_num<signed char>(0x45, 0x45);
  test_num<char8_t>(0xAB, 0xAB);
  test_num<char16_t>(0xABCD, 0xCDAB);
  test_num<char32_t>(0xABCDEF01, 0x01EFCDAB);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_implementation_defined_size<wchar_t>();
#endif

  test_implementation_defined_size<short>();
  test_implementation_defined_size<unsigned short>();
  test_implementation_defined_size<int>();
  test_implementation_defined_size<unsigned int>();
  test_implementation_defined_size<long>();
  test_implementation_defined_size<unsigned long>();
  test_implementation_defined_size<long long>();
  test_implementation_defined_size<unsigned long long>();

#if TEST_HAS_EXTENSION(bit_int)
  // _BitInt(N) where digits + is_signed == sizeof * CHAR_BIT (no padding
  // bits) is accepted; other widths are rejected by the static_assert
  // inside the function body (see byteswap.verify.cpp).

  // sizeof == 1
  test_num<unsigned _BitInt(8)>(0xAB, 0xAB);
  test_num<signed _BitInt(8)>(0x12, 0x12);

  // sizeof == 2: __builtin_bswap16 fallback or __builtin_bswapg
  test_num<unsigned _BitInt(16)>(0xCDEF, 0xEFCD);
  test_num<signed _BitInt(16)>(0x1234, 0x3412);

  // sizeof == 4: __builtin_bswap32 fallback or __builtin_bswapg
  test_num<unsigned _BitInt(32)>(0x01234567U, 0x67452301U);
  test_num<signed _BitInt(32)>(0x01234567, 0x67452301);

  // sizeof == 8: __builtin_bswap64 fallback or __builtin_bswapg
  test_num<unsigned _BitInt(64)>(0x0123456789ABCDEFULL, 0xEFCDAB8967452301ULL);
  test_num<signed _BitInt(64)>(0x0123456789ABCDEFLL, static_cast<signed _BitInt(64)>(0xEFCDAB8967452301ULL));

#  if __BITINT_MAXWIDTH__ >= 128
  // sizeof == 16: __builtin_bswap128 fallback or __builtin_bswapg.
  unsigned _BitInt(128) v128 =
      (static_cast<unsigned _BitInt(128)>(0x0123456789ABCDEFULL) << 64) |
      static_cast<unsigned _BitInt(128)>(0x13579BDF02468ACEULL);
  unsigned _BitInt(128) v128_swapped =
      (static_cast<unsigned _BitInt(128)>(0xCE8A4602DF9B5713ULL) << 64) |
      static_cast<unsigned _BitInt(128)>(0xEFCDAB8967452301ULL);
  test_num<unsigned _BitInt(128)>(v128, v128_swapped);
  test_num<signed _BitInt(128)>(static_cast<signed _BitInt(128)>(v128), static_cast<signed _BitInt(128)>(v128_swapped));
#  endif

#  if __has_builtin(__builtin_bswapg) && __BITINT_MAXWIDTH__ >= 256
  // sizeof > 16: only the __builtin_bswapg path supports widths beyond what
  // __builtin_bswap16/32/64/128 cover.
  unsigned _BitInt(256) v256 =
      (static_cast<unsigned _BitInt(256)>(0xDEADBEEFCAFEBABEULL) << 128) |
      (static_cast<unsigned _BitInt(256)>(0x1234567890ABCDEFULL) << 64) |
      static_cast<unsigned _BitInt(256)>(0xFEDCBA9876543210ULL);
  assert(std::byteswap(std::byteswap(v256)) == v256);
  ASSERT_SAME_TYPE(decltype(std::byteswap(v256)), unsigned _BitInt(256));
  ASSERT_NOEXCEPT(std::byteswap(v256));

  // Spot check: low byte of input must end up as the high byte of the output.
  unsigned _BitInt(256) lo_only = 0xAB;
  auto lo_swapped               = std::byteswap(lo_only);
  assert(static_cast<unsigned char>(lo_swapped >> ((sizeof(lo_swapped) - 1) * CHAR_BIT)) == 0xAB);

  // Mid-value test: a distinct byte at every position so an off-by-one in the
  // generated code surfaces directly.
  unsigned _BitInt(256) ramp = 0;
  for (int i = 0; i < 32; ++i)
    ramp |= static_cast<unsigned _BitInt(256)>(i) << (i * CHAR_BIT);
  auto ramp_swapped = std::byteswap(ramp);
  for (int i = 0; i < 32; ++i)
    assert(static_cast<unsigned char>(ramp_swapped >> ((31 - i) * CHAR_BIT)) == i);
#  endif
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
