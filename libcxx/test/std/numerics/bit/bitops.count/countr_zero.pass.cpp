//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr int countr_zero(T x) noexcept;

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
    ASSERT_SAME_TYPE(decltype(std::countr_zero(T())), int);
    ASSERT_NOEXCEPT(std::countr_zero(T()));
    T max = std::numeric_limits<T>::max();

    assert(std::countr_zero(T(0)) == std::numeric_limits<T>::digits);
    assert(std::countr_zero(T(1)) == 0);
    assert(std::countr_zero(T(2)) == 1);
    assert(std::countr_zero(T(3)) == 0);
    assert(std::countr_zero(T(4)) == 2);
    assert(std::countr_zero(T(5)) == 0);
    assert(std::countr_zero(T(6)) == 1);
    assert(std::countr_zero(T(7)) == 0);
    assert(std::countr_zero(T(8)) == 3);
    assert(std::countr_zero(T(9)) == 0);
    assert(std::countr_zero(T(126)) == 1);
    assert(std::countr_zero(T(127)) == 0);
    assert(std::countr_zero(T(128)) == 7);
    assert(std::countr_zero(T(129)) == 0);
    assert(std::countr_zero(T(130)) == 1);
    assert(std::countr_zero(max) == 0);

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = T(128) << 32;
        assert(std::countr_zero(val-1) ==  0);
        assert(std::countr_zero(val)   == 39);
        assert(std::countr_zero(val+1) ==  0);
        val <<= 60;
        assert(std::countr_zero(val-1) ==  0);
        assert(std::countr_zero(val)   == 99);
        assert(std::countr_zero(val+1) ==  0);
    }
#endif

    return true;
}

int main(int, char**)
{
    {
    auto lambda = [](auto x) -> decltype(std::countr_zero(x)) {};
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
      assert(std::countr_zero(T8(0)) == 8);
      assert(std::countr_zero(T8(1)) == 0);
      assert(std::countr_zero(T8(2)) == 1);
      assert(std::countr_zero(T8(3)) == 0);
      assert(std::countr_zero(T8(4)) == 2);
      assert(std::countr_zero(T8(8)) == 3);
      assert(std::countr_zero(T8(128)) == 7);
      assert(std::countr_zero(T8(~T8(0))) == 0);
      assert(std::countr_zero(T32(0)) == 32);
      assert(std::countr_zero(T32(1)) == 0);
      assert(std::countr_zero(T32(2)) == 1);
      assert(std::countr_zero(T32(4)) == 2);
      assert(std::countr_zero(T32(126)) == 1);
      assert(std::countr_zero(T32(128)) == 7);
      assert(std::countr_zero(T32(1) << 31) == 31);
      assert(std::countr_zero(T64(0)) == 64);
      assert(std::countr_zero(T64(1)) == 0);
      assert(std::countr_zero(T64(1) << 63) == 63);

      // Odd widths: safe for nonzero inputs only.
      assert(std::countr_zero(T13(1)) == 0);
      assert(std::countr_zero(T13(2)) == 1);
      assert(std::countr_zero(T13(4)) == 2);
      assert(std::countr_zero(T13(128)) == 7);
      assert(std::countr_zero(T13(1) << 12) == 12);
    }
#  if __BITINT_MAXWIDTH__ >= 128
    {
      using T77  = unsigned _BitInt(77);
      using T128 = unsigned _BitInt(128);
      assert(std::countr_zero(T77(1)) == 0);
      assert(std::countr_zero(T77(2)) == 1);
      assert(std::countr_zero(T77(1) << 76) == 76);

      assert(std::countr_zero(T128(0)) == 128);
      assert(std::countr_zero(T128(1)) == 0);
      assert(std::countr_zero(T128(T128(1) << 63)) == 63);
      assert(std::countr_zero(T128(T128(1) << 64)) == 64);
      assert(std::countr_zero(T128(1) << 127) == 127);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    {
      using T129 = unsigned _BitInt(129);
      using T256 = unsigned _BitInt(256);
      assert(std::countr_zero(T129(1) << 128) == 128);

      assert(std::countr_zero(T256(1)) == 0);
      assert(std::countr_zero(T256(1) << 127) == 127);
      assert(std::countr_zero(T256(1) << 128) == 128);
      assert(std::countr_zero(T256(1) << 200) == 200);
      assert(std::countr_zero(T256(1) << 255) == 255);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 4096
    {
      using T4096 = unsigned _BitInt(4096);
      assert(std::countr_zero(T4096(1)) == 0);
      assert(std::countr_zero(T4096(1) << 2048) == 2048);
      assert(std::countr_zero(T4096(1) << 4095) == 4095);
    }
#  endif
#endif // TEST_HAS_EXTENSION(bit_int)

    return 0;
}
