//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr T bit_ceil(T x) noexcept;

// Constraints: T is an unsigned integer type
// Returns: The minimal value y such that has_single_bit(y) is true and y >= x;
//    if y is not representable as a value of type T, the result is an unspecified value.

#include <bit>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "test_macros.h"

struct A {};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <class T>
constexpr bool test()
{
    ASSERT_SAME_TYPE(decltype(std::bit_ceil(T())), T);
    LIBCPP_ASSERT_NOEXCEPT(std::bit_ceil(T()));

    assert(std::bit_ceil(T(0)) == T(1));
    assert(std::bit_ceil(T(1)) == T(1));
    assert(std::bit_ceil(T(2)) == T(2));
    assert(std::bit_ceil(T(3)) == T(4));
    assert(std::bit_ceil(T(4)) == T(4));
    assert(std::bit_ceil(T(5)) == T(8));
    assert(std::bit_ceil(T(6)) == T(8));
    assert(std::bit_ceil(T(7)) == T(8));
    assert(std::bit_ceil(T(8)) == T(8));
    assert(std::bit_ceil(T(9)) == T(16));
    assert(std::bit_ceil(T(60)) == T(64));
    assert(std::bit_ceil(T(61)) == T(64));
    assert(std::bit_ceil(T(62)) == T(64));
    assert(std::bit_ceil(T(63)) == T(64));
    assert(std::bit_ceil(T(64)) == T(64));
    assert(std::bit_ceil(T(65)) == T(128));
    assert(std::bit_ceil(T(66)) == T(128));
    assert(std::bit_ceil(T(67)) == T(128));
    assert(std::bit_ceil(T(68)) == T(128));
    assert(std::bit_ceil(T(69)) == T(128));

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = 168;
        T ceil = 256;

        assert(std::bit_ceil(val) == ceil);
        assert(std::bit_ceil(val << 32) == (ceil << 32));
        assert(std::bit_ceil((val << 64) | 0x1) == (ceil << 64));
        assert(std::bit_ceil((val << 72) | 0x1) == (ceil << 72));
        assert(std::bit_ceil((val << 100) | 0x1) == (ceil << 100));
    }
#endif

    return true;
}

int main(int, char**)
{
    {
    auto lambda = [](auto x) -> decltype(std::bit_ceil(x)) {};
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
