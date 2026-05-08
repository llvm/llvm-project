//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr int countl_one(T x) noexcept;

// Constraints: T is an unsigned integer type
// The number of consecutive 1 bits, starting from the most significant bit.
//   [ Note: Returns N if x == std::numeric_limits<T>::max(). ]

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
    ASSERT_SAME_TYPE(decltype(std::countl_one(T())), int);
    ASSERT_NOEXCEPT(std::countl_one(T()));
    T max = std::numeric_limits<T>::max();

    assert(std::countl_one(T(0)) == 0);
    assert(std::countl_one(T(1)) == 0);
    assert(std::countl_one(T(10)) == 0);
    assert(std::countl_one(T(100)) == 0);
    assert(std::countl_one(max) == std::numeric_limits<T>::digits);
    assert(std::countl_one(T(max - 1)) == std::numeric_limits<T>::digits - 1);
    assert(std::countl_one(T(max - 2)) == std::numeric_limits<T>::digits - 2);
    assert(std::countl_one(T(max - 3)) == std::numeric_limits<T>::digits - 2);
    assert(std::countl_one(T(max - 4)) == std::numeric_limits<T>::digits - 3);
    assert(std::countl_one(T(max - 5)) == std::numeric_limits<T>::digits - 3);
    assert(std::countl_one(T(max - 6)) == std::numeric_limits<T>::digits - 3);
    assert(std::countl_one(T(max - 7)) == std::numeric_limits<T>::digits - 3);
    assert(std::countl_one(T(max - 8)) == std::numeric_limits<T>::digits - 4);
    assert(std::countl_one(T(max - 9)) == std::numeric_limits<T>::digits - 4);
    assert(std::countl_one(T(max - 126)) == std::numeric_limits<T>::digits - 7);
    assert(std::countl_one(T(max - 127)) == std::numeric_limits<T>::digits - 7);
    assert(std::countl_one(T(max - 128)) == std::numeric_limits<T>::digits - 8);

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = 128;
        assert(std::countl_one(~val) == 120);
        val <<= 32;
        assert(std::countl_one(~val) == 88);
        val <<= 60;
        assert(std::countl_one(~val) == 28);
    }
#endif

    return true;
}

int main(int, char**)
{
    {
    auto lambda = [](auto x) -> decltype(std::countl_one(x)) {};
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

    // _BitInt tests. Width tiers follow C23 7.18.2.5.
#if TEST_HAS_EXTENSION(bit_int)
    {
      using T13 = unsigned _BitInt(13);
      using T32 = unsigned _BitInt(32);
      using T64 = unsigned _BitInt(64);

      // Byte-aligned widths: numeric_limits::digits is correct, so all
      // values including all-ones are safe to test.
      assert(std::countl_one(T32(0)) == 0);
      assert(std::countl_one(T32(1)) == 0);
      assert(std::countl_one(T32(~T32(0))) == 32);
      assert(std::countl_one(T32(~T32(0) - 1)) == 31);
      assert(std::countl_one(T32(~T32(0) - 2)) == 30);
      assert(std::countl_one(T32(~T32(0) - 8)) == 28);
      assert(std::countl_one(T32(~T32(0) - 127)) == 25);
      assert(std::countl_one(T32(~T32(0) - 128)) == 24);
      assert(std::countl_one(T64(0)) == 0);
      assert(std::countl_one(T64(~T64(0))) == 64);
      assert(std::countl_one(T64(~T64(0) - 1)) == 63);

      // Odd widths: safe for values that are not all-ones.
      assert(std::countl_one(T13(0)) == 0);
      assert(std::countl_one(T13(1)) == 0);
      assert(std::countl_one(T13(~T13(0) - 1)) == 12);
      assert(std::countl_one(T13(~T13(0) - 2)) == 11);
    }
#  if __BITINT_MAXWIDTH__ >= 128
    {
      using T77  = unsigned _BitInt(77);
      using T128 = unsigned _BitInt(128);
      assert(std::countl_one(T77(0)) == 0);
      assert(std::countl_one(T77(1)) == 0);
      assert(std::countl_one(T77(~T77(0) - 1)) == 76);

      assert(std::countl_one(T128(0)) == 0);
      assert(std::countl_one(T128(~T128(0))) == 128);
      assert(std::countl_one(T128(~T128(0) - 1)) == 127);
      // Clear a single bit at position 64: 63 leading ones.
      assert(std::countl_one(T128(~T128(0) ^ (T128(1) << 64))) == 63);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    {
      using T256 = unsigned _BitInt(256);
      assert(std::countl_one(T256(0)) == 0);
      assert(std::countl_one(T256(~T256(0))) == 256);
      assert(std::countl_one(T256(~T256(0) - 1)) == 255);
      // Clear a single bit at position 100: 155 leading ones.
      assert(std::countl_one(T256(~T256(0) ^ (T256(1) << 100))) == 155);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 4096
    {
      using T4096 = unsigned _BitInt(4096);
      assert(std::countl_one(T4096(0)) == 0);
      assert(std::countl_one(T4096(~T4096(0))) == 4096);
      // Clear a single bit at position 1000: 3095 leading ones.
      assert(std::countl_one(T4096(~T4096(0) ^ (T4096(1) << 1000))) == 3095);
    }
#  endif
#endif // TEST_HAS_EXTENSION(bit_int)

    return 0;
}
