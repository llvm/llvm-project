//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr int rotl(T x, unsigned int s) noexcept;

// Constraints: T is an unsigned integer type

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
    ASSERT_SAME_TYPE(decltype(std::rotl(T(), 0)), T);
    ASSERT_NOEXCEPT(std::rotl(T(), 0));
    T max = std::numeric_limits<T>::max();
    T highbit = std::rotr(T(1), 1);

    assert(std::rotl(T(max - 1), 0) == T(max - 1));
    assert(std::rotl(T(max - 1), 1) == T(max - 2));
    assert(std::rotl(T(max - 1), 2) == T(max - 4));
    assert(std::rotl(T(max - 1), 3) == T(max - 8));
    assert(std::rotl(T(max - 1), 4) == T(max - 16));
    assert(std::rotl(T(max - 1), 5) == T(max - 32));
    assert(std::rotl(T(max - 1), 6) == T(max - 64));
    assert(std::rotl(T(max - 1), 7) == T(max - 128));
    assert(std::rotl(T(max - 1), std::numeric_limits<int>::max()) ==
           std::rotl(T(max - 1), std::numeric_limits<int>::max() % std::numeric_limits<T>::digits));

    assert(std::rotl(T(max - 1), -1) == T(max - highbit));
    assert(std::rotl(T(max - 1), -2) == T(max - (highbit >> 1)));
    assert(std::rotl(T(max - 1), -3) == T(max - (highbit >> 2)));
    assert(std::rotl(T(max - 1), -4) == T(max - (highbit >> 3)));
    assert(std::rotl(T(max - 1), -5) == T(max - (highbit >> 4)));
    assert(std::rotl(T(max - 1), -6) == T(max - (highbit >> 5)));
    assert(std::rotl(T(max - 1), -7) == T(max - (highbit >> 6)));
    assert(std::rotl(T(max - 1), std::numeric_limits<int>::min()) ==
           std::rotl(T(max - 1), std::numeric_limits<int>::min() % std::numeric_limits<T>::digits));

    assert(std::rotl(T(1), 0) == T(1));
    assert(std::rotl(T(1), 1) == T(2));
    assert(std::rotl(T(1), 2) == T(4));
    assert(std::rotl(T(1), 3) == T(8));
    assert(std::rotl(T(1), 4) == T(16));
    assert(std::rotl(T(1), 5) == T(32));
    assert(std::rotl(T(1), 6) == T(64));
    assert(std::rotl(T(1), 7) == T(128));

    assert(std::rotl(T(128), -1) == T(64));
    assert(std::rotl(T(128), -2) == T(32));
    assert(std::rotl(T(128), -3) == T(16));
    assert(std::rotl(T(128), -4) == T(8));
    assert(std::rotl(T(128), -5) == T(4));
    assert(std::rotl(T(128), -6) == T(2));
    assert(std::rotl(T(128), -7) == T(1));

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = (T(1) << 63) | (T(1) << 64);
        assert(std::rotl(val, 0) == val);
        assert(std::rotl(val, 128) == val);
        assert(std::rotl(val, 256) == val);
        assert(std::rotl(val, 1) == val << 1);
        assert(std::rotl(val, 127) == val >> 1);
        assert(std::rotl(T(3), 127) == ((T(1) << 127) | T(1)));

        assert(std::rotl(val, -128) == val);
        assert(std::rotl(val, -256) == val);
        assert(std::rotl(val, -1) == val >> 1);
        assert(std::rotl(val, -127) == val << 1);
        assert(std::rotl(T(3), -1) == ((T(1) << 127) | T(1)));
    }
#endif

    return true;
}

int main(int, char**)
{
    {
    auto lambda = [](auto x) -> decltype(std::rotl(x, 1U)) {};
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
    // rotl uses numeric_limits::digits internally, so only byte-aligned
    // widths are safe (where digits matches the actual bit width).
#if TEST_HAS_EXTENSION(bit_int)
    {
      using T32 = unsigned _BitInt(32);
      using T64 = unsigned _BitInt(64);

      T32 m32 = ~T32(0);
      assert(std::rotl(T32(1), 0) == T32(1));
      assert(std::rotl(T32(1), 1) == T32(2));
      assert(std::rotl(T32(1), 4) == T32(16));
      assert(std::rotl(T32(1), 7) == T32(128));
      assert(std::rotl(T32(128), -1) == T32(64));
      assert(std::rotl(T32(128), -7) == T32(1));
      assert(std::rotl(T32(m32 - 1), 0) == T32(m32 - 1));
      assert(std::rotl(T32(m32 - 1), 1) == T32(m32 - 2));
      assert(std::rotl(T32(m32 - 1), 4) == T32(m32 - 16));
      // Full rotation returns original.
      assert(std::rotl(T32(1), 32) == T32(1));

      assert(std::rotl(T64(1), 0) == T64(1));
      assert(std::rotl(T64(1), 4) == T64(16));
      assert(std::rotl(T64(1), -1) == (T64(1) << 63));
      assert(std::rotl(T64(1), 64) == T64(1));
    }
#  if __BITINT_MAXWIDTH__ >= 128
    {
      using T128 = unsigned _BitInt(128);
      assert(std::rotl(T128(1), 0) == T128(1));
      assert(std::rotl(T128(1), 4) == T128(16));
      assert(std::rotl(T128(1), 63) == (T128(1) << 63));
      assert(std::rotl(T128(1), 64) == (T128(1) << 64));
      assert(std::rotl(T128(1), -1) == (T128(1) << 127));
      assert(std::rotl(T128(1), 128) == T128(1));
      // Multi-bit wrap-around across limb boundary.
      assert(std::rotl(T128(3) << 62, 4) == T128(3) << 66);
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    {
      using T256 = unsigned _BitInt(256);
      assert(std::rotl(T256(1), 0) == T256(1));
      assert(std::rotl(T256(1), 4) == T256(16));
      assert(std::rotl(T256(1), 200) == (T256(1) << 200));
      assert(std::rotl(T256(1), -1) == (T256(1) << 255));
      assert(std::rotl(T256(1), 256) == T256(1));
      assert(std::rotl(T256(~T256(0) - 1), 1) == T256(~T256(0) - 2));
      // Wrap-around: rotate a high bit to low.
      assert(std::rotl(T256(1) << 255, 1) == T256(1));
      // Modulo: rotation amount larger than width.
      assert(std::rotl(T256(1), 256 + 4) == T256(1) << 4);
    }
#  endif
#endif // TEST_HAS_EXTENSION(bit_int)

    return 0;
}
