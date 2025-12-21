//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr int bit_width(T x) noexcept;

// Constraints: T is an unsigned integer type
// Returns: If x == 0, 0; otherwise one plus the base-2 logarithm of x, with any fractional part discarded.


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
    ASSERT_SAME_TYPE(decltype(std::bit_width(T())), int);
    ASSERT_NOEXCEPT(std::bit_width(T()));
    T max = std::numeric_limits<T>::max();

    assert(std::bit_width(T(0)) == 0);
    assert(std::bit_width(T(1)) == 1);
    assert(std::bit_width(T(2)) == 2);
    assert(std::bit_width(T(3)) == 2);
    assert(std::bit_width(T(4)) == 3);
    assert(std::bit_width(T(5)) == 3);
    assert(std::bit_width(T(6)) == 3);
    assert(std::bit_width(T(7)) == 3);
    assert(std::bit_width(T(8)) == 4);
    assert(std::bit_width(T(9)) == 4);
    assert(std::bit_width(T(125)) == 7);
    assert(std::bit_width(T(126)) == 7);
    assert(std::bit_width(T(127)) == 7);
    assert(std::bit_width(T(128)) == 8);
    assert(std::bit_width(T(129)) == 8);
    assert(std::bit_width(T(130)) == 8);
    assert(std::bit_width(T(max - 1)) == std::numeric_limits<T>::digits);
    assert(std::bit_width(max) == std::numeric_limits<T>::digits);

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = 128;
        val <<= 32;
        assert(std::bit_width(val-1) == 39);
        assert(std::bit_width(val)   == 40);
        assert(std::bit_width(val+1) == 40);
        val <<= 60;
        assert(std::bit_width(val-1) == 99);
        assert(std::bit_width(val)   == 100);
        assert(std::bit_width(val+1) == 100);
    }
#endif

    return true;
}

int main(int, char**)
{

    {
    auto lambda = [](auto x) -> decltype(std::bit_width(x)) {};
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

    return 0;
}
