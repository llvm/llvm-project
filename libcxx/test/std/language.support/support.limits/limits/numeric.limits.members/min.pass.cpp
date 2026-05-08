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

    // _BitInt(N): min is 0 for unsigned and -2^(N-1) for signed. The shift
    // `1 << digits` flowed through the buggy digits field, so this also
    // exercises the digits fix for non-byte-aligned widths.
#if TEST_HAS_EXTENSION(bit_int)
    test<unsigned _BitInt(8)>(0);
    test<signed _BitInt(8)>(-(signed _BitInt(8))(1 << 7));
    test<unsigned _BitInt(13)>(0);
    test<signed _BitInt(13)>(-(signed _BitInt(13))(1 << 12));
    test<unsigned _BitInt(64)>(0);
    test<signed _BitInt(64)>(-(signed _BitInt(64))(1ULL << 63));
#  if __BITINT_MAXWIDTH__ >= 128
    test<unsigned _BitInt(77)>(0);
    test<signed _BitInt(77)>(-((signed _BitInt(77))1 << 76));
    test<unsigned _BitInt(128)>(0);
    test<signed _BitInt(128)>(-((signed _BitInt(128))1 << 127));
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    test<unsigned _BitInt(256)>(0);
    test<signed _BitInt(256)>(-((signed _BitInt(256))1 << 255));
#  endif
#endif

    return 0;
}
