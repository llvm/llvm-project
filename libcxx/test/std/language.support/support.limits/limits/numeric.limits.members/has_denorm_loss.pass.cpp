//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// has_denorm_loss

#include <limits>

#include "test_macros.h"

template <class T, bool expected>
void
test()
{
    static_assert(std::numeric_limits<T>::has_denorm_loss == expected, "has_denorm_loss test 1");
    static_assert(std::numeric_limits<const T>::has_denorm_loss == expected, "has_denorm_loss test 2");
    static_assert(std::numeric_limits<volatile T>::has_denorm_loss == expected, "has_denorm_loss test 3");
    static_assert(std::numeric_limits<const volatile T>::has_denorm_loss == expected, "has_denorm_loss test 4");
}

int main(int, char**)
{
    test<bool, false>();
    test<char, false>();
    test<signed char, false>();
    test<unsigned char, false>();
    test<wchar_t, false>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, false>();
#endif
#ifndef TEST_HAS_NO_UNICODE_CHARS
    test<char16_t, false>();
    test<char32_t, false>();
#endif
    test<short, false>();
    test<unsigned short, false>();
    test<int, false>();
    test<unsigned int, false>();
    test<long, false>();
    test<unsigned long, false>();
    test<long long, false>();
    test<unsigned long long, false>();
#ifndef TEST_HAS_NO_INT128
    test<__int128_t, false>();
    test<__uint128_t, false>();
#endif
    test<float, false>();
    test<double, false>();
    test<long double, false>();

  return 0;
}
