//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr bool has_single_bit(T x) noexcept;

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
    ASSERT_SAME_TYPE(decltype(std::has_single_bit(T())), bool);
    ASSERT_NOEXCEPT(std::has_single_bit(T()));
    T max = std::numeric_limits<T>::max();

    assert(!std::has_single_bit(T(0)));
    assert( std::has_single_bit(T(1)));
    assert( std::has_single_bit(T(2)));
    assert(!std::has_single_bit(T(3)));
    assert( std::has_single_bit(T(4)));
    assert(!std::has_single_bit(T(5)));
    assert(!std::has_single_bit(T(6)));
    assert(!std::has_single_bit(T(7)));
    assert( std::has_single_bit(T(8)));
    assert(!std::has_single_bit(T(9)));
    assert(!std::has_single_bit(T(127)));
    assert( std::has_single_bit(T(128)));
    assert(!std::has_single_bit(T(129)));
    assert(!std::has_single_bit(max));

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = T(1) << 32;
        assert(!std::has_single_bit(val-1));
        assert( std::has_single_bit(val));
        assert(!std::has_single_bit(val+1));
        val <<= 60;
        assert(!std::has_single_bit(val-1));
        assert( std::has_single_bit(val));
        assert(!std::has_single_bit(val+1));

        T x = (T(1) << 63);
        T y = (T(1) << 64);
        assert( std::has_single_bit(x));
        assert( std::has_single_bit(y));
        assert(!std::has_single_bit(x + y));
    }
#endif

    return true;
}

int main(int, char**)
{
    {
    auto lambda = [](auto x) -> decltype(std::has_single_bit(x)) {};
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

      assert(!std::has_single_bit(T32(0)));
      assert(std::has_single_bit(T32(1)));
      assert(std::has_single_bit(T32(2)));
      assert(!std::has_single_bit(T32(3)));
      assert(std::has_single_bit(T32(4)));
      assert(!std::has_single_bit(T32(5)));
      assert(!std::has_single_bit(T32(6)));
      assert(!std::has_single_bit(T32(7)));
      assert(std::has_single_bit(T32(8)));
      assert(!std::has_single_bit(T32(9)));
      assert(std::has_single_bit(T32(128)));
      assert(!std::has_single_bit(T32(127)));
      assert(!std::has_single_bit(T32(129)));
      assert(!std::has_single_bit(T32(~T32(0))));
      assert(!std::has_single_bit(T64(0)));
      assert(std::has_single_bit(T64(1)));
      assert(std::has_single_bit(T64(T64(1) << 32)));
      assert(std::has_single_bit(T64(T64(1) << 63)));
      assert(!std::has_single_bit(T64(~T64(0))));

      // Odd widths: has_single_bit has no digits dependency.
      assert(!std::has_single_bit(T13(0)));
      assert(std::has_single_bit(T13(1)));
      assert(std::has_single_bit(T13(2)));
      assert(!std::has_single_bit(T13(3)));
      assert(std::has_single_bit(T13(4)));
      assert(std::has_single_bit(T13(64)));
      assert(!std::has_single_bit(T13(65)));
      assert(!std::has_single_bit(T13(~T13(0))));
    }
#  if __BITINT_MAXWIDTH__ >= 128
    {
      using T77  = unsigned _BitInt(77);
      using T128 = unsigned _BitInt(128);
      assert(!std::has_single_bit(T77(0)));
      assert(std::has_single_bit(T77(1)));
      assert(std::has_single_bit(T77(2)));
      assert(!std::has_single_bit(T77(3)));
      assert(std::has_single_bit(T77(T77(1) << 76)));
      assert(!std::has_single_bit(T77((T77(1) << 76) | T77(1))));
      assert(!std::has_single_bit(T77(~T77(0))));

      assert(!std::has_single_bit(T128(0)));
      assert(std::has_single_bit(T128(1)));
      assert(std::has_single_bit(T128(T128(1) << 64)));
      assert(std::has_single_bit(T128(T128(1) << 127)));
      assert(!std::has_single_bit(T128(~T128(0))));
      // Two bits: definitely not a single bit.
      assert(!std::has_single_bit(T128((T128(1) << 127) | T128(1))));
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    {
      using T129 = unsigned _BitInt(129);
      using T256 = unsigned _BitInt(256);
      assert(std::has_single_bit(T129(1) << 128));
      assert(!std::has_single_bit(T129(~T129(0))));

      assert(!std::has_single_bit(T256(0)));
      assert(std::has_single_bit(T256(1)));
      assert(std::has_single_bit(T256(1) << 200));
      assert(std::has_single_bit(T256(1) << 255));
      assert(!std::has_single_bit((T256(1) << 200) | T256(1)));
      assert(!std::has_single_bit(T256(~T256(0))));
      assert(!std::has_single_bit(T256(~T256(0) / 3))); // 0x5555... = 128 bits
    }
#  endif
#  if __BITINT_MAXWIDTH__ >= 4096
    {
      using T4096 = unsigned _BitInt(4096);
      assert(!std::has_single_bit(T4096(0)));
      assert(std::has_single_bit(T4096(1)));
      assert(std::has_single_bit(T4096(1) << 4095));
      assert(std::has_single_bit(T4096(1) << 2048));
      assert(!std::has_single_bit(T4096(~T4096(0))));
      assert(!std::has_single_bit((T4096(1) << 4095) | T4096(1)));
    }
#  endif
#endif // TEST_HAS_EXTENSION(bit_int)

    return 0;
}
