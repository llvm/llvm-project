//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// radix

#include <limits>
#include <cfloat>

#include "test_macros.h"

template <class T, int expected>
void
test()
{
    static_assert(std::numeric_limits<T>::radix == expected, "radix test 1");
    static_assert(std::numeric_limits<const T>::radix == expected, "radix test 2");
    static_assert(std::numeric_limits<volatile T>::radix == expected, "radix test 3");
    static_assert(std::numeric_limits<const volatile T>::radix == expected, "radix test 4");
}

int main(int, char**)
{
    test<bool, 2>();
    test<char, 2>();
    test<signed char, 2>();
    test<unsigned char, 2>();
    test<wchar_t, 2>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, 2>();
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t, 2>();
    test<char32_t, 2>();
#endif
    test<short, 2>();
    test<unsigned short, 2>();
    test<int, 2>();
    test<unsigned int, 2>();
    test<long, 2>();
    test<unsigned long, 2>();
    test<long long, 2>();
    test<unsigned long long, 2>();
#ifndef TEST_HAS_NO_INT128
    test<__int128_t, 2>();
    test<__uint128_t, 2>();
#endif
    test<float, FLT_RADIX>();
    test<double, FLT_RADIX>();
    test<long double, FLT_RADIX>();

  return 0;
}
