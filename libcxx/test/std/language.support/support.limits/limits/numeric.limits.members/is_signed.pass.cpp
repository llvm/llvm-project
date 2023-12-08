//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// is_signed

#include <limits>

#include "test_macros.h"

template <class T, bool expected>
void
test()
{
    static_assert(std::numeric_limits<T>::is_signed == expected, "is_signed test 1");
    static_assert(std::numeric_limits<const T>::is_signed == expected, "is_signed test 2");
    static_assert(std::numeric_limits<volatile T>::is_signed == expected, "is_signed test 3");
    static_assert(std::numeric_limits<const volatile T>::is_signed == expected, "is_signed test 4");
}

int main(int, char**)
{
    test<bool, false>();
    test<char, char(-1) < char(0)>();
    test<signed char, true>();
    test<unsigned char, false>();
    test<wchar_t, wchar_t(-1) < wchar_t(0)>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, false>();
#endif
    test<char16_t, false>();
    test<char32_t, false>();
    test<short, true>();
    test<unsigned short, false>();
    test<int, true>();
    test<unsigned int, false>();
    test<long, true>();
    test<unsigned long, false>();
    test<long long, true>();
    test<unsigned long long, false>();
#ifndef TEST_HAS_NO_INT128
    test<__int128_t, true>();
    test<__uint128_t, false>();
#endif
    test<float, true>();
    test<double, true>();
    test<long double, true>();

  return 0;
}
