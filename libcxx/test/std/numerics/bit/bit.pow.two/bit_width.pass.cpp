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
//   constexpr T bit_width(T x) noexcept;

// Constraints: T is an unsigned integer type
// Returns: If x == 0, 0; otherwise one plus the base-2 logarithm of x, with any fractional part discarded.


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
    ASSERT_SAME_TYPE(decltype(std::bit_width(T())), T);
    ASSERT_NOEXCEPT(std::bit_width(T()));
    T max = std::numeric_limits<T>::max();

    assert(std::bit_width(T(0)) == T(0));
    assert(std::bit_width(T(1)) == T(1));
    assert(std::bit_width(T(2)) == T(2));
    assert(std::bit_width(T(3)) == T(2));
    assert(std::bit_width(T(4)) == T(3));
    assert(std::bit_width(T(5)) == T(3));
    assert(std::bit_width(T(6)) == T(3));
    assert(std::bit_width(T(7)) == T(3));
    assert(std::bit_width(T(8)) == T(4));
    assert(std::bit_width(T(9)) == T(4));
    assert(std::bit_width(T(125)) == T(7));
    assert(std::bit_width(T(126)) == T(7));
    assert(std::bit_width(T(127)) == T(7));
    assert(std::bit_width(T(128)) == T(8));
    assert(std::bit_width(T(129)) == T(8));
    assert(std::bit_width(T(130)) == T(8));
    assert(std::bit_width(T(max - 1)) == T(std::numeric_limits<T>::digits));
    assert(std::bit_width(max) == T(std::numeric_limits<T>::digits));

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
