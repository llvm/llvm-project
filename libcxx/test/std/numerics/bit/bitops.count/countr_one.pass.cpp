//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr int countr_one(T x) noexcept;

// Constraints: T is an unsigned integer type
// Returns: The number of consecutive 1 bits, starting from the least significant bit.
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
    ASSERT_SAME_TYPE(decltype(std::countr_one(T())), int);
    ASSERT_NOEXCEPT(std::countr_one(T()));
    T max = std::numeric_limits<T>::max();

    assert(std::countr_one(T(0)) == 0);
    assert(std::countr_one(T(1)) == 1);
    assert(std::countr_one(T(2)) == 0);
    assert(std::countr_one(T(3)) == 2);
    assert(std::countr_one(T(4)) == 0);
    assert(std::countr_one(T(5)) == 1);
    assert(std::countr_one(T(6)) == 0);
    assert(std::countr_one(T(7)) == 3);
    assert(std::countr_one(T(8)) == 0);
    assert(std::countr_one(T(9)) == 1);
    assert(std::countr_one(T(126)) == 0);
    assert(std::countr_one(T(127)) == 7);
    assert(std::countr_one(T(128)) == 0);
    assert(std::countr_one(T(max - 1)) == 0);
    assert(std::countr_one(max) == std::numeric_limits<T>::digits);

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = 128;
        assert(std::countr_one(val-1) ==  7);
        assert(std::countr_one(val)   ==  0);
        assert(std::countr_one(val+1) ==  1);
        val <<= 32;
        assert(std::countr_one(val-1) == 39);
        assert(std::countr_one(val)   ==  0);
        assert(std::countr_one(val+1) ==  1);
        val <<= 60;
        assert(std::countr_one(val-1) == 99);
        assert(std::countr_one(val)   ==  0);
        assert(std::countr_one(val+1) ==  1);
    }
#endif

    return true;
}

int main(int, char**)
{
    {
    auto lambda = [](auto x) -> decltype(std::countr_one(x)) {};
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
      assert(std::countr_one(T32(0)) == 0);
      assert(std::countr_one(T32(1)) == 1);
      assert(std::countr_one(T32(2)) == 0);
      assert(std::countr_one(T32(3)) == 2);
      assert(std::countr_one(T32(4)) == 0);
      assert(std::countr_one(T32(5)) == 1);
      assert(std::countr_one(T32(7)) == 3);
      assert(std::countr_one(T32(15)) == 4);
      assert(std::countr_one(T32(127)) == 7);
      assert(std::countr_one(T32(128)) == 0);
      assert(std::countr_one(T32(~T32(0) - 1)) == 0);
      assert(std::countr_one(T32(~T32(0))) == 32);
      assert(std::countr_one(T64(0)) == 0);
      assert(std::countr_one(T64(1)) == 1);
      assert(std::countr_one(T64(7)) == 3);
      assert(std::countr_one(T64(~T64(0))) == 64);

      // Odd widths: safe for values that are not all-ones.
      assert(std::countr_one(T13(0)) == 0);
      assert(std::countr_one(T13(1)) == 1);
      assert(std::countr_one(T13(3)) == 2);
      assert(std::countr_one(T13(7)) == 3);
      assert(std::countr_one(T13(15)) == 4);
      assert(std::countr_one(T13(127)) == 7);
      assert(std::countr_one(T13(128)) == 0);
    }
#  if __BITINT_MAXWIDTH__ >= 128
    {
      using T77  = unsigned _BitInt(77);
      using T128 = unsigned _BitInt(128);
      assert(std::countr_one(T77(0)) == 0);
      assert(std::countr_one(T77(1)) == 1);
      assert(std::countr_one(T77(3)) == 2);
      assert(std::countr_one(T77(7)) == 3);
      assert(std::countr_one(T77(127)) == 7);

      assert(std::countr_one(T128(0)) == 0);
      assert(std::countr_one(T128(1)) == 1);
      assert(std::countr_one(T128(~T128(0))) == 128);
      // Mask of low 64 bits: 64 trailing ones, then a zero.
      assert(std::countr_one(T128((T128(1) << 64) - 1)) == 64);
      // Mask of low 65 bits: 65 trailing ones (spans 64-bit boundary).
      assert(std::countr_one(T128((T128(1) << 65) - 1)) == 65);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    {
      using T256 = unsigned _BitInt(256);
      assert(std::countr_one(T256(0)) == 0);
      assert(std::countr_one(T256(~T256(0))) == 256);
      // Mask of low 128 bits: 128 trailing ones.
      assert(std::countr_one(T256((T256(1) << 128) - 1)) == 128);
      // Mask of low 200 bits: 200 trailing ones.
      assert(std::countr_one(T256((T256(1) << 200) - 1)) == 200);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 4096
    {
      using T4096 = unsigned _BitInt(4096);
      assert(std::countr_one(T4096(0)) == 0);
      assert(std::countr_one(T4096(~T4096(0))) == 4096);
      // Mask of low 1000 bits: 1000 trailing ones.
      assert(std::countr_one(T4096((T4096(1) << 1000) - 1)) == 1000);
    }
#  endif
#endif // TEST_HAS_EXTENSION(bit_int)

    return 0;
}
