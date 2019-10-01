//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// has_denorm

#include <limits>

#include "test_macros.h"

template <class T, std::float_denorm_style expected>
void
test()
{
    static_assert(std::numeric_limits<T>::has_denorm == expected, "has_denorm test 1");
    static_assert(std::numeric_limits<const T>::has_denorm == expected, "has_denorm test 2");
    static_assert(std::numeric_limits<volatile T>::has_denorm == expected, "has_denorm test 3");
    static_assert(std::numeric_limits<const volatile T>::has_denorm == expected, "has_denorm test 4");
}

int main(int, char**)
{
    test<bool, std::denorm_absent>();
    test<char, std::denorm_absent>();
    test<signed char, std::denorm_absent>();
    test<unsigned char, std::denorm_absent>();
    test<wchar_t, std::denorm_absent>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, std::denorm_absent>();
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t, std::denorm_absent>();
    test<char32_t, std::denorm_absent>();
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
    test<short, std::denorm_absent>();
    test<unsigned short, std::denorm_absent>();
    test<int, std::denorm_absent>();
    test<unsigned int, std::denorm_absent>();
    test<long, std::denorm_absent>();
    test<unsigned long, std::denorm_absent>();
    test<long long, std::denorm_absent>();
    test<unsigned long long, std::denorm_absent>();
#ifndef _LIBCPP_HAS_NO_INT128
    test<__int128_t, std::denorm_absent>();
    test<__uint128_t, std::denorm_absent>();
#endif
    test<float, std::denorm_present>();
    test<double, std::denorm_present>();
    test<long double, std::denorm_present>();

  return 0;
}
