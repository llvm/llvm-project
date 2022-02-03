//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// infinity()

#include <limits>
#include <cfloat>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test(T expected)
{
    assert(std::numeric_limits<T>::infinity() == expected);
    assert(std::numeric_limits<const T>::infinity() == expected);
    assert(std::numeric_limits<volatile T>::infinity() == expected);
    assert(std::numeric_limits<const volatile T>::infinity() == expected);
}

extern float zero;

int main(int, char**)
{
    test<bool>(false);
    test<char>(0);
    test<signed char>(0);
    test<unsigned char>(0);
    test<wchar_t>(0);
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>(0);
#endif
#ifndef TEST_HAS_NO_UNICODE_CHARS
    test<char16_t>(0);
    test<char32_t>(0);
#endif
    test<short>(0);
    test<unsigned short>(0);
    test<int>(0);
    test<unsigned int>(0);
    test<long>(0);
    test<unsigned long>(0);
    test<long long>(0);
    test<unsigned long long>(0);
#ifndef TEST_HAS_NO_INT128
    test<__int128_t>(0);
    test<__uint128_t>(0);
#endif
    test<float>(1.f/zero);
    test<double>(1./zero);
    test<long double>(1./zero);

  return 0;
}

float zero = 0;
