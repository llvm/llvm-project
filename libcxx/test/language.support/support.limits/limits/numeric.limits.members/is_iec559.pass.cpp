//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// is_iec559

#include <limits>

template <class T, bool expected>
void
test()
{
    static_assert(std::numeric_limits<T>::is_iec559 == expected, "is_iec559 test 1");
    static_assert(std::numeric_limits<const T>::is_iec559 == expected, "is_iec559 test 2");
    static_assert(std::numeric_limits<volatile T>::is_iec559 == expected, "is_iec559 test 3");
    static_assert(std::numeric_limits<const volatile T>::is_iec559 == expected, "is_iec559 test 4");
}

int main()
{
    test<bool, false>();
    test<char, false>();
    test<signed char, false>();
    test<unsigned char, false>();
    test<wchar_t, false>();
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t, false>();
    test<char32_t, false>();
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
    test<short, false>();
    test<unsigned short, false>();
    test<int, false>();
    test<unsigned int, false>();
    test<long, false>();
    test<unsigned long, false>();
    test<long long, false>();
    test<unsigned long long, false>();
    test<float, true>();
    test<double, true>();
    test<long double, true>();
}
