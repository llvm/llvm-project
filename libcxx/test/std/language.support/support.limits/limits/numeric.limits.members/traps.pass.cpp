//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// traps

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <limits>

#include "test_macros.h"

#if defined(__i386__) || defined(__x86_64__) || defined(__wasm__)
static const bool integral_types_trap = true;
#else
static const bool integral_types_trap = false;
#endif

template <class T, bool expected>
void
test()
{
    static_assert(std::numeric_limits<T>::traps == expected, "traps test 1");
    static_assert(std::numeric_limits<const T>::traps == expected, "traps test 2");
    static_assert(std::numeric_limits<volatile T>::traps == expected, "traps test 3");
    static_assert(std::numeric_limits<const volatile T>::traps == expected, "traps test 4");
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
    test<char16_t, false>();
    test<char32_t, false>();
    test<short, false>();
    test<unsigned short, false>();
    test<int, integral_types_trap>();
    test<unsigned int, integral_types_trap>();
    test<long, integral_types_trap>();
    test<unsigned long, integral_types_trap>();
    test<long long, integral_types_trap>();
    test<unsigned long long, integral_types_trap>();
#ifndef TEST_HAS_NO_INT128
    test<__int128_t, integral_types_trap>();
    test<__uint128_t, integral_types_trap>();
#endif
    test<float, false>();
    test<double, false>();
    test<long double, false>();

  return 0;
}
