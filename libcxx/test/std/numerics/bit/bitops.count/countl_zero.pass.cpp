//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr int countl_zero(T x) noexcept;

// Constraints: T is an unsigned integer type
// Returns: The number of consecutive 0 bits, starting from the most significant bit.
//   [ Note: Returns N if x == 0. ]

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
    ASSERT_SAME_TYPE(decltype(std::countl_zero(T())), int);
    ASSERT_NOEXCEPT(std::countl_zero(T()));
    T max = std::numeric_limits<T>::max();
    int dig = std::numeric_limits<T>::digits;

    assert(std::countl_zero(T(0)) == dig);
    assert(std::countl_zero(T(1)) == dig - 1);
    assert(std::countl_zero(T(2)) == dig - 2);
    assert(std::countl_zero(T(3)) == dig - 2);
    assert(std::countl_zero(T(4)) == dig - 3);
    assert(std::countl_zero(T(5)) == dig - 3);
    assert(std::countl_zero(T(6)) == dig - 3);
    assert(std::countl_zero(T(7)) == dig - 3);
    assert(std::countl_zero(T(8)) == dig - 4);
    assert(std::countl_zero(T(9)) == dig - 4);
    assert(std::countl_zero(T(127)) == dig - 7);
    assert(std::countl_zero(T(128)) == dig - 8);
    assert(std::countl_zero(max) == 0);

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = T(128) << 32;
        assert(std::countl_zero(val-1) == 89);
        assert(std::countl_zero(val)   == 88);
        assert(std::countl_zero(val+1) == 88);
        val <<= 60;
        assert(std::countl_zero(val-1) == 29);
        assert(std::countl_zero(val)   == 28);
        assert(std::countl_zero(val+1) == 28);
    }
#endif

    return true;
}

int main(int, char**)
{
    {
    auto lambda = [](auto x) -> decltype(std::countl_zero(x)) {};
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
      using T8  = unsigned _BitInt(8);
      using T13 = unsigned _BitInt(13);
      using T32 = unsigned _BitInt(32);
      using T64 = unsigned _BitInt(64);

      // Byte-aligned widths: numeric_limits::digits is correct, so all
      // values including zero are safe to test.
      assert(std::countl_zero(T8(0)) == 8);
      assert(std::countl_zero(T8(1)) == 7);
      assert(std::countl_zero(T8(2)) == 6);
      assert(std::countl_zero(T8(3)) == 6);
      assert(std::countl_zero(T8(4)) == 5);
      assert(std::countl_zero(T8(8)) == 4);
      assert(std::countl_zero(T8(127)) == 1);
      assert(std::countl_zero(T8(128)) == 0);
      assert(std::countl_zero(T8(~T8(0))) == 0);
      assert(std::countl_zero(T32(0)) == 32);
      assert(std::countl_zero(T32(1)) == 31);
      assert(std::countl_zero(T32(2)) == 30);
      assert(std::countl_zero(T32(3)) == 30);
      assert(std::countl_zero(T32(127)) == 25);
      assert(std::countl_zero(T32(128)) == 24);
      assert(std::countl_zero(T32(~T32(0))) == 0);
      assert(std::countl_zero(T64(0)) == 64);
      assert(std::countl_zero(T64(1)) == 63);
      assert(std::countl_zero(T64(T64(1) << 63)) == 0);
      assert(std::countl_zero(T64(~T64(0))) == 0);

      // Odd widths: safe for nonzero inputs only (digits is the fallback
      // for zero via __builtin_clzg).
      assert(std::countl_zero(T13(1)) == 12);
      assert(std::countl_zero(T13(2)) == 11);
      assert(std::countl_zero(T13(3)) == 11);
      assert(std::countl_zero(T13(127)) == 6);
      assert(std::countl_zero(T13(128)) == 5);
      assert(std::countl_zero(T13(~T13(0))) == 0);
    }
#  if __BITINT_MAXWIDTH__ >= 128
    {
      using T77  = unsigned _BitInt(77);
      using T128 = unsigned _BitInt(128);
      assert(std::countl_zero(T77(1)) == 76);
      assert(std::countl_zero(T77(T77(1) << 76)) == 0);
      assert(std::countl_zero(T77(~T77(0))) == 0);

      assert(std::countl_zero(T128(0)) == 128);
      assert(std::countl_zero(T128(1)) == 127);
      assert(std::countl_zero(T128(T128(1) << 64)) == 63);
      assert(std::countl_zero(T128(T128(1) << 127)) == 0);
      assert(std::countl_zero(T128(~T128(0))) == 0);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    {
      using T129 = unsigned _BitInt(129);
      using T256 = unsigned _BitInt(256);
      // Odd width around 128-bit limb boundary.
      assert(std::countl_zero(T129(1)) == 128);
      assert(std::countl_zero(T129(1) << 128) == 0);
      assert(std::countl_zero(T129(~T129(0))) == 0);

      assert(std::countl_zero(T256(~T256(0))) == 0);
      assert(std::countl_zero(T256(1)) == 255);
      // Bit set at position 200: 55 leading zeros.
      assert(std::countl_zero(T256(1) << 200) == 55);
      // Bit at position 127 (just below 128-bit boundary): 128 leading zeros.
      assert(std::countl_zero(T256(1) << 127) == 128);
      // Bit at position 128 (just at 128-bit boundary): 127 leading zeros.
      assert(std::countl_zero(T256(1) << 128) == 127);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 4096
    {
      using T4096 = unsigned _BitInt(4096);
      assert(std::countl_zero(T4096(1)) == 4095);
      assert(std::countl_zero(T4096(1) << 4095) == 0);
      assert(std::countl_zero(T4096(1) << 2048) == 2047);
      assert(std::countl_zero(T4096(~T4096(0))) == 0);
    }
#  endif
#endif // TEST_HAS_EXTENSION(bit_int)

    return 0;
}
