//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// min()

#include <limits>
#include <climits>
#include <cfloat>
#include <cassert>

#include "test_macros.h"

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
#   include <cwchar>
#endif

template <class T>
void
test(T expected)
{
    assert(std::numeric_limits<T>::min() == expected);
    assert(std::numeric_limits<T>::is_bounded || !std::numeric_limits<T>::is_signed);
    assert(std::numeric_limits<const T>::min() == expected);
    assert(std::numeric_limits<const T>::is_bounded || !std::numeric_limits<const T>::is_signed);
    assert(std::numeric_limits<volatile T>::min() == expected);
    assert(std::numeric_limits<volatile T>::is_bounded || !std::numeric_limits<volatile T>::is_signed);
    assert(std::numeric_limits<const volatile T>::min() == expected);
    assert(std::numeric_limits<const volatile T>::is_bounded || !std::numeric_limits<const volatile T>::is_signed);
}

int main(int, char**)
{
    test<bool>(false);
    test<char>(CHAR_MIN);
    test<signed char>(SCHAR_MIN);
    test<unsigned char>(0);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>(WCHAR_MIN);
#endif
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>(0);
#endif
    test<char16_t>(0);
    test<char32_t>(0);
    test<short>(SHRT_MIN);
    test<unsigned short>(0);
    test<int>(INT_MIN);
    test<unsigned int>(0);
    test<long>(LONG_MIN);
    test<unsigned long>(0);
    test<long long>(LLONG_MIN);
    test<unsigned long long>(0);
#ifndef TEST_HAS_NO_INT128
    test<__int128_t>(-__int128_t(__uint128_t(-1)/2) - 1);
    test<__uint128_t>(0);
#endif
    test<float>(FLT_MIN);
    test<double>(DBL_MIN);
    test<long double>(LDBL_MIN);

  return 0;
}
