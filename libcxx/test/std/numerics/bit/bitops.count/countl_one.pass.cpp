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
//   constexpr int countl_one(T x) noexcept;

// Constraints: T is an unsigned integer type
// The number of consecutive 1 bits, starting from the most significant bit.
//   [ Note: Returns N if x == std::numeric_limits<T>::max(). ]

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

    static_assert(!std::is_invocable_v<L, int8_t>);
    static_assert(!std::is_invocable_v<L, int16_t>);
    static_assert(!std::is_invocable_v<L, int32_t>);
    static_assert(!std::is_invocable_v<L, int64_t>);
    static_assert(!std::is_invocable_v<L, intmax_t>);
    static_assert(!std::is_invocable_v<L, intptr_t>);
    static_assert(!std::is_invocable_v<L, ptrdiff_t>);

    static_assert(!std::is_invocable_v<L, bool>);
    static_assert(!std::is_invocable_v<L, char>);
    static_assert(!std::is_invocable_v<L, wchar_t>);
#ifndef _LIBCPP_HAS_NO_CHAR8_T
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
    static_assert(test<uint8_t>());
    static_assert(test<uint16_t>());
    static_assert(test<uint32_t>());
    static_assert(test<uint64_t>());
    static_assert(test<uintmax_t>());
    static_assert(test<uintptr_t>());
    static_assert(test<size_t>());

    test<unsigned char>();
    test<unsigned short>();
    test<unsigned int>();
    test<unsigned long>();
    test<unsigned long long>();
#ifndef TEST_HAS_NO_INT128
    test<__uint128_t>();
#endif
    test<uint8_t>();
    test<uint16_t>();
    test<uint32_t>();
    test<uint64_t>();
    test<uintmax_t>();
    test<uintptr_t>();
    test<size_t>();

    return 0;
}

