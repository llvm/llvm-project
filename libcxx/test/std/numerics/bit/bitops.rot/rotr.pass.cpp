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
//   constexpr int rotr(T x, unsigned int s) noexcept;

// Constraints: T is an unsigned integer type

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
    ASSERT_SAME_TYPE(decltype(std::rotr(T(), 0)), T);
    ASSERT_NOEXCEPT(std::rotr(T(), 0));
    T max = std::numeric_limits<T>::max();
    T highbit = std::rotr(T(1), 1);

    assert(std::rotr(T(max - 1), 0) == T(max - 1));
    assert(std::rotr(T(max - 1), 1) == T(max - highbit));
    assert(std::rotr(T(max - 1), 2) == T(max - (highbit >> 1)));
    assert(std::rotr(T(max - 1), 3) == T(max - (highbit >> 2)));
    assert(std::rotr(T(max - 1), 4) == T(max - (highbit >> 3)));
    assert(std::rotr(T(max - 1), 5) == T(max - (highbit >> 4)));
    assert(std::rotr(T(max - 1), 6) == T(max - (highbit >> 5)));
    assert(std::rotr(T(max - 1), 7) == T(max - (highbit >> 6)));

    assert(std::rotr(T(max - 1), -1) == T(max - 2));
    assert(std::rotr(T(max - 1), -2) == T(max - 4));
    assert(std::rotr(T(max - 1), -3) == T(max - 8));
    assert(std::rotr(T(max - 1), -4) == T(max - 16));
    assert(std::rotr(T(max - 1), -5) == T(max - 32));
    assert(std::rotr(T(max - 1), -6) == T(max - 64));
    assert(std::rotr(T(max - 1), -7) == T(max - 128));

    assert(std::rotr(T(128), 0) == T(128));
    assert(std::rotr(T(128), 1) == T(64));
    assert(std::rotr(T(128), 2) == T(32));
    assert(std::rotr(T(128), 3) == T(16));
    assert(std::rotr(T(128), 4) == T(8));
    assert(std::rotr(T(128), 5) == T(4));
    assert(std::rotr(T(128), 6) == T(2));
    assert(std::rotr(T(128), 7) == T(1));

    assert(std::rotr(T(1), -1) == T(2));
    assert(std::rotr(T(1), -2) == T(4));
    assert(std::rotr(T(1), -3) == T(8));
    assert(std::rotr(T(1), -4) == T(16));
    assert(std::rotr(T(1), -5) == T(32));
    assert(std::rotr(T(1), -6) == T(64));
    assert(std::rotr(T(1), -7) == T(128));

#ifndef TEST_HAS_NO_INT128
    if constexpr (std::is_same_v<T, __uint128_t>) {
        T val = (T(1) << 63) | (T(1) << 64);
        assert(std::rotr(val, 0) == val);
        assert(std::rotr(val, 128) == val);
        assert(std::rotr(val, 256) == val);
        assert(std::rotr(val, 1) == val >> 1);
        assert(std::rotr(val, 127) == val << 1);
        assert(std::rotr(T(3), 1) == ((T(1) << 127) | T(1)));

        assert(std::rotr(val, -128) == val);
        assert(std::rotr(val, -256) == val);
        assert(std::rotr(val, -1) == val << 1);
        assert(std::rotr(val, -127) == val >> 1);
        assert(std::rotr(T(3), -127) == ((T(1) << 127) | T(1)));
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

    return 0;
}
