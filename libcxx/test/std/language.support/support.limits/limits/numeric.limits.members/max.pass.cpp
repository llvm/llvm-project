//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// max()

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
    assert(std::numeric_limits<T>::max() == expected);
    assert(std::numeric_limits<T>::is_bounded);
    assert(std::numeric_limits<const T>::max() == expected);
    assert(std::numeric_limits<const T>::is_bounded);
    assert(std::numeric_limits<volatile T>::max() == expected);
    assert(std::numeric_limits<volatile T>::is_bounded);
    assert(std::numeric_limits<const volatile T>::max() == expected);
    assert(std::numeric_limits<const volatile T>::is_bounded);
}

int main(int, char**)
{
    test<bool>(true);
    test<char>(CHAR_MAX);
    test<signed char>(SCHAR_MAX);
    test<unsigned char>(UCHAR_MAX);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>(WCHAR_MAX);
#endif
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>(UCHAR_MAX); // ??
#endif
    test<char16_t>(USHRT_MAX);
    test<char32_t>(UINT_MAX);
    test<short>(SHRT_MAX);
    test<unsigned short>(USHRT_MAX);
    test<int>(INT_MAX);
    test<unsigned int>(UINT_MAX);
    test<long>(LONG_MAX);
    test<unsigned long>(ULONG_MAX);
    test<long long>(LLONG_MAX);
    test<unsigned long long>(ULLONG_MAX);
#ifndef TEST_HAS_NO_INT128
    test<__int128_t>(__int128_t(__uint128_t(-1)/2));
    test<__uint128_t>(__uint128_t(-1));
#endif
    test<float>(FLT_MAX);
    test<double>(DBL_MAX);
    test<long double>(LDBL_MAX);

    // _BitInt(N): max is 2^N - 1 for unsigned and 2^(N-1) - 1 for signed.
    // Exercises the digits fix through `__max = ~0 ^ __min`.
#if TEST_HAS_EXTENSION(bit_int)
    test<unsigned _BitInt(8)>((unsigned _BitInt(8)) ~(unsigned _BitInt(8))0);
    test<signed _BitInt(8)>((signed _BitInt(8))0x7F);
    test<unsigned _BitInt(13)>((unsigned _BitInt(13))0x1FFF);
    test<signed _BitInt(13)>((signed _BitInt(13))0x0FFF);
    test<unsigned _BitInt(64)>((unsigned _BitInt(64)) ~(unsigned _BitInt(64))0);
    test<signed _BitInt(64)>((signed _BitInt(64))0x7FFFFFFFFFFFFFFFLL);
#  if __BITINT_MAXWIDTH__ >= 128
    test<unsigned _BitInt(77)>((unsigned _BitInt(77)) ~(unsigned _BitInt(77))0);
    test<signed _BitInt(77)>((signed _BitInt(77)) ~((signed _BitInt(77))1 << 76));
    test<unsigned _BitInt(128)>((unsigned _BitInt(128)) ~(unsigned _BitInt(128))0);
    test<signed _BitInt(128)>((signed _BitInt(128)) ~((signed _BitInt(128))1 << 127));
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    test<unsigned _BitInt(256)>((unsigned _BitInt(256)) ~(unsigned _BitInt(256))0);
    test<signed _BitInt(256)>((signed _BitInt(256)) ~((signed _BitInt(256))1 << 255));
#  endif
#endif

    return 0;
}
