//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// Specializations shall be provided for each arithmetic type, both floating
// point and integer, including bool. The member is_specialized shall be
// true for all such specializations of numeric_limits.

// Non-arithmetic standard types, such as complex<T> (26.3.2), shall not
// have specializations.

// From [numeric.limits]:

// The value of each member of a specialization of numeric_limits on a cv
// -qualified type cv T shall be equal to the value of the corresponding
// member of the specialization on the unqualified type T.

// More convenient to test it here.

#include <limits>
#include <complex>

#include "test_macros.h"

template <class T>
void test()
{
    static_assert(std::numeric_limits<T>::is_specialized,
                 "std::numeric_limits<T>::is_specialized");
    static_assert(std::numeric_limits<const T>::is_specialized,
                 "std::numeric_limits<const T>::is_specialized");
    static_assert(std::numeric_limits<volatile T>::is_specialized,
                 "std::numeric_limits<volatile T>::is_specialized");
    static_assert(std::numeric_limits<const volatile T>::is_specialized,
                 "std::numeric_limits<const volatile T>::is_specialized");
}

int main(int, char**)
{
    test<bool>();
    test<char>();
    test<wchar_t>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>();
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t>();
    test<char32_t>();
#endif
    test<signed char>();
    test<unsigned char>();
    test<signed short>();
    test<unsigned short>();
    test<signed int>();
    test<unsigned int>();
    test<signed long>();
    test<unsigned long>();
    test<signed long long>();
    test<unsigned long long>();
#ifndef _LIBCPP_HAS_NO_INT128
    test<__int128_t>();
    test<__uint128_t>();
#endif
    test<float>();
    test<double>();
    test<long double>();
    static_assert(!std::numeric_limits<std::complex<double> >::is_specialized,
                 "!std::numeric_limits<std::complex<double> >::is_specialized");

  return 0;
}
