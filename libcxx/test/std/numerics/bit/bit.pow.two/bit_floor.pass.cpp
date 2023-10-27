//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr T bit_floor(T x) noexcept;

// Constraints: T is an unsigned integer type
// Returns: If x == 0, 0; otherwise the maximal value y such that bit_floor(y) is true and y <= x.

#include <bit>
#include <cassert>
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
    ASSERT_SAME_TYPE(decltype(std::bit_floor(T())), T);
    LIBCPP_ASSERT_NOEXCEPT(std::bit_floor(T()));
    T max = std::numeric_limits<T>::max();

    assert(std::bit_floor(T(0)) == T(0));
    assert(std::bit_floor(T(1)) == T(1));
    assert(std::bit_floor(T(2)) == T(2));
    assert(std::bit_floor(T(3)) == T(2));
    assert(std::bit_floor(T(4)) == T(4));
    assert(std::bit_floor(T(5)) == T(4));
    assert(std::bit_floor(T(6)) == T(4));
    assert(std::bit_floor(T(7)) == T(4));
    assert(std::bit_floor(T(8)) == T(8));
    assert(std::bit_floor(T(9)) == T(8));
    assert(std::bit_floor(T(125)) == T(64));
    assert(std::bit_floor(T(126)) == T(64));
    assert(std::bit_floor(T(127)) == T(64));
    assert(std::bit_floor(T(128)) == T(128));
    assert(std::bit_floor(T(129)) == T(128));
    assert(std::bit_floor(max) == T(max - (max >> 1)));

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = T(128) << 32;
        assert(std::bit_floor(val-1) == val/2);
        assert(std::bit_floor(val)   == val);
        assert(std::bit_floor(val+1) == val);
        val <<= 60;
        assert(std::bit_floor(val-1) == val/2);
        assert(std::bit_floor(val)   == val);
        assert(std::bit_floor(val+1) == val);
    }
#endif

    return true;
}

int main(int, char**)
{
    {
    auto lambda = [](auto x) -> decltype(std::bit_floor(x)) {};
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
