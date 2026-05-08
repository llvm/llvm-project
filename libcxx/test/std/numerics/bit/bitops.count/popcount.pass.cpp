//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr int popcount(T x) noexcept;

// Constraints: T is an unsigned integer type
// Returns: The number of bits set to one in the value of x.

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "test_macros.h"

struct A {};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <class T>
constexpr bool test()
{
    ASSERT_SAME_TYPE(decltype(std::popcount(T())), int);
    ASSERT_NOEXCEPT(std::popcount(T()));
    T max = std::numeric_limits<T>::max();

    assert(std::popcount(T(0)) == 0);
    assert(std::popcount(T(1)) == 1);
    assert(std::popcount(T(2)) == 1);
    assert(std::popcount(T(3)) == 2);
    assert(std::popcount(T(4)) == 1);
    assert(std::popcount(T(5)) == 2);
    assert(std::popcount(T(6)) == 2);
    assert(std::popcount(T(7)) == 3);
    assert(std::popcount(T(8)) == 1);
    assert(std::popcount(T(9)) == 2);
    assert(std::popcount(T(121)) == 5);
    assert(std::popcount(T(127)) == 7);
    assert(std::popcount(T(128)) == 1);
    assert(std::popcount(T(130)) == 2);
    assert(std::popcount(T(max >> 1)) == std::numeric_limits<T>::digits - 1);
    assert(std::popcount(T(max - 1)) == std::numeric_limits<T>::digits - 1);
    assert(std::popcount(max) == std::numeric_limits<T>::digits);

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = 128;
        assert(std::popcount(val-1) ==  7);
        assert(std::popcount(val)   ==  1);
        assert(std::popcount(val+1) ==  2);
        val <<= 32;
        assert(std::popcount(val-1) == 39);
        assert(std::popcount(val)   ==  1);
        assert(std::popcount(val+1) ==  2);
        val <<= 60;
        assert(std::popcount(val-1) == 99);
        assert(std::popcount(val)   ==  1);
        assert(std::popcount(val+1) ==  2);

        T x = T(1) << 63;
        T y = T(1) << 64;
        assert(std::popcount(x) == 1);
        assert(std::popcount(y) == 1);
        assert(std::popcount(x+y) == 2);
    }
#endif

    return true;
}

int main(int, char**)
{
    {
    auto lambda = [](auto x) -> decltype(std::popcount(x)) {};
    using L = decltype(lambda);

    static_assert(!std::is_invocable_v<L, signed char>);
    static_assert(!std::is_invocable_v<L, short>);
    static_assert(!std::is_invocable_v<L, int>);
    static_assert(!std::is_invocable_v<L, long>);
    static_assert(!std::is_invocable_v<L, long long>);
#ifndef TEST_HAS_NO_INT128
    static_assert(!std::is_invocable_v<L, __int128_t>);
#endif

    static_assert(!std::is_invocable_v<L, std::int8_t>);
    static_assert(!std::is_invocable_v<L, std::int16_t>);
    static_assert(!std::is_invocable_v<L, std::int32_t>);
    static_assert(!std::is_invocable_v<L, std::int64_t>);
    static_assert(!std::is_invocable_v<L, std::intmax_t>);
    static_assert(!std::is_invocable_v<L, std::intptr_t>);
    static_assert(!std::is_invocable_v<L, std::ptrdiff_t>);

    static_assert(!std::is_invocable_v<L, bool>);
    static_assert(!std::is_invocable_v<L, char>);
    static_assert(!std::is_invocable_v<L, wchar_t>);
#ifndef TEST_HAS_NO_CHAR8_T
    static_assert(!std::is_invocable_v<L, char8_t>);
#endif
    static_assert(!std::is_invocable_v<L, char16_t>);
    static_assert(!std::is_invocable_v<L, char32_t>);

    static_assert(!std::is_invocable_v<L, A>);
    static_assert(!std::is_invocable_v<L, A*>);
    static_assert(!std::is_invocable_v<L, E1>);
    static_assert(!std::is_invocable_v<L, E2>);
    }

    static_assert(test<unsigned char>());
    static_assert(test<unsigned short>());
    static_assert(test<unsigned int>());
    static_assert(test<unsigned long>());
    static_assert(test<unsigned long long>());
#ifndef TEST_HAS_NO_INT128
    static_assert(test<__uint128_t>());
#endif
    static_assert(test<std::uint8_t>());
    static_assert(test<std::uint16_t>());
    static_assert(test<std::uint32_t>());
    static_assert(test<std::uint64_t>());
    static_assert(test<std::uintmax_t>());
    static_assert(test<std::uintptr_t>());
    static_assert(test<std::size_t>());

    test<unsigned char>();
    test<unsigned short>();
    test<unsigned int>();
    test<unsigned long>();
    test<unsigned long long>();
#ifndef TEST_HAS_NO_INT128
    test<__uint128_t>();
#endif
    test<std::uint8_t>();
    test<std::uint16_t>();
    test<std::uint32_t>();
    test<std::uint64_t>();
    test<std::uintmax_t>();
    test<std::uintptr_t>();
    test<std::size_t>();

    // _BitInt tests. Width tiers follow C23 7.18.2.5: BITINT_MAXWIDTH is
    // guaranteed to be >= ULLONG_WIDTH (>= 64). Anything beyond that is
    // optional and must be guarded by __BITINT_MAXWIDTH__.
#if TEST_HAS_EXTENSION(bit_int)
    {
      // Guaranteed widths (<= 64 bits).
      using T8  = unsigned _BitInt(8);
      using T13 = unsigned _BitInt(13);
      using T32 = unsigned _BitInt(32);
      using T64 = unsigned _BitInt(64);

      assert(std::popcount(T8(0)) == 0);
      assert(std::popcount(T8(1)) == 1);
      assert(std::popcount(T8(2)) == 1);
      assert(std::popcount(T8(3)) == 2);
      assert(std::popcount(T8(7)) == 3);
      assert(std::popcount(T8(0x55)) == 4);
      assert(std::popcount(T8(0xFF)) == 8);

      assert(std::popcount(T32(0)) == 0);
      assert(std::popcount(T32(1)) == 1);
      assert(std::popcount(T32(3)) == 2);
      assert(std::popcount(T32(127)) == 7);
      assert(std::popcount(T32(128)) == 1);
      assert(std::popcount(T32(130)) == 2);
      assert(std::popcount(T32(~T32(0))) == 32);

      assert(std::popcount(T64(0)) == 0);
      assert(std::popcount(T64(1)) == 1);
      assert(std::popcount(T64(127)) == 7);
      assert(std::popcount(T64(~T64(0))) == 64);
      assert(std::popcount(T64(~T64(0) >> 1)) == 63);

      // Odd (non-byte-aligned) widths: popcount has no digits dependency.
      assert(std::popcount(T13(0)) == 0);
      assert(std::popcount(T13(1)) == 1);
      assert(std::popcount(T13(3)) == 2);
      assert(std::popcount(T13(7)) == 3);
      assert(std::popcount(T13(127)) == 7);
      assert(std::popcount(T13(128)) == 1);
      assert(std::popcount(T13(~T13(0))) == 13);
    }
#  if __BITINT_MAXWIDTH__ >= 128
    {
      using T77  = unsigned _BitInt(77);
      using T128 = unsigned _BitInt(128);
      assert(std::popcount(T77(0)) == 0);
      assert(std::popcount(T77(1)) == 1);
      assert(std::popcount(T77(3)) == 2);
      assert(std::popcount(T77(127)) == 7);
      assert(std::popcount(T77(~T77(0))) == 77);
      assert(std::popcount(T77(~T77(0) - 1)) == 76);

      assert(std::popcount(T128(0)) == 0);
      assert(std::popcount(T128(1)) == 1);
      assert(std::popcount(T128(~T128(0))) == 128);
      assert(std::popcount(T128(~T128(0) - 1)) == 127);
      // Alternating bit pattern: ~0 / 3 == 0x5555...5555 in any width.
      assert(std::popcount(T128(~T128(0) / 3)) == 64);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    {
      using T129 = unsigned _BitInt(129);
      using T255 = unsigned _BitInt(255);
      using T256 = unsigned _BitInt(256);

      // Odd widths at 128-bit boundary.
      assert(std::popcount(T129(0)) == 0);
      assert(std::popcount(T129(~T129(0))) == 129);
      assert(std::popcount(T129(1) << 128) == 1);
      assert(std::popcount(T255(~T255(0))) == 255);
      assert(std::popcount(T255(1) << 254) == 1);

      assert(std::popcount(T256(0)) == 0);
      assert(std::popcount(T256(~T256(0))) == 256);
      // Alternating bit pattern: ~0 / 3 == 0x5555...5555 (128 bits set).
      assert(std::popcount(T256(~T256(0) / 3)) == 128);
      // (1 << 200) - 1 has exactly 200 bits set.
      T256 mask200 = T256(1) << 200;
      mask200 -= 1;
      assert(std::popcount(mask200) == 200);
      // Single high bit at position 255.
      assert(std::popcount(T256(1) << 255) == 1);
      // Two bits spanning the low/high halves.
      assert(std::popcount(T256(1) | (T256(1) << 255)) == 2);
      // Exactly 4 bits at positions 0, 64, 128, 255.
      T256 scattered = T256(1) | (T256(1) << 64) | (T256(1) << 128) | (T256(1) << 255);
      assert(std::popcount(scattered) == 4);
      // All ones minus a single bit.
      assert(std::popcount(T256(~T256(0)) ^ (T256(1) << 200)) == 255);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 4096
    {
      // Huge width exercises multi-limb iteration.
      using T4096 = unsigned _BitInt(4096);
      assert(std::popcount(T4096(0)) == 0);
      assert(std::popcount(T4096(~T4096(0))) == 4096);
      assert(std::popcount(T4096(~T4096(0) / 3)) == 2048); // alternating bits
      assert(std::popcount(T4096(1) << 4095) == 1);
      // 1000 bits set starting from position 0.
      T4096 mask1000 = T4096(1) << 1000;
      mask1000 -= 1;
      assert(std::popcount(mask1000) == 1000);
    }
#  endif
#endif // TEST_HAS_EXTENSION(bit_int)

    return 0;
}
