//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// template <class T>
// struct hash
//     : public unary_function<T, size_t>
// {
//     size_t operator()(T val) const;
// };

#include <functional>
#include <cassert>
#include <type_traits>
#include <cstddef>
#include <limits>

#include "test_macros.h"

template <class T>
void
test()
{
    typedef std::hash<T> H;
#if TEST_STD_VER <= 17
    static_assert((std::is_same<typename H::argument_type, T>::value), "");
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "");
#endif
    ASSERT_NOEXCEPT(H()(T()));
    H h;

    for (int i = 0; i <= 5; ++i)
    {
        T t(static_cast<T>(i));
        const bool small = std::integral_constant<bool, sizeof(T) <= sizeof(std::size_t)>::value; // avoid compiler warnings
        if (small)
        {
            const std::size_t result = h(t);
            LIBCPP_ASSERT(result == static_cast<std::size_t>(t));
            ((void)result); // Prevent unused warning
        }
    }
}

int main(int, char**)
{
    test<bool>();
    test<char>();
    test<signed char>();
    test<unsigned char>();
    test<char16_t>();
    test<char32_t>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>();
#endif
    test<short>();
    test<unsigned short>();
    test<int>();
    test<unsigned int>();
    test<long>();
    test<unsigned long>();
    test<long long>();
    test<unsigned long long>();

//  LWG #2119
    test<std::ptrdiff_t>();
    test<std::size_t>();

    test<std::int8_t>();
    test<std::int16_t>();
    test<std::int32_t>();
    test<std::int64_t>();

    test<std::int_fast8_t>();
    test<std::int_fast16_t>();
    test<std::int_fast32_t>();
    test<std::int_fast64_t>();

    test<std::int_least8_t>();
    test<std::int_least16_t>();
    test<std::int_least32_t>();
    test<std::int_least64_t>();

    test<std::intmax_t>();
    test<std::intptr_t>();

    test<std::uint8_t>();
    test<std::uint16_t>();
    test<std::uint32_t>();
    test<std::uint64_t>();

    test<std::uint_fast8_t>();
    test<std::uint_fast16_t>();
    test<std::uint_fast32_t>();
    test<std::uint_fast64_t>();

    test<std::uint_least8_t>();
    test<std::uint_least16_t>();
    test<std::uint_least32_t>();
    test<std::uint_least64_t>();

    test<std::uintmax_t>();
    test<std::uintptr_t>();

#ifndef TEST_HAS_NO_INT128
    test<__int128_t>();
    test<__uint128_t>();
#endif

  return 0;
}
