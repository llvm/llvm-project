//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// max_exponent

#include <limits>
#include <cfloat>

template <class T, int expected>
void
test()
{
    static_assert(std::numeric_limits<T>::max_exponent == expected, "max_exponent test 1");
    static_assert(std::numeric_limits<const T>::max_exponent == expected, "max_exponent test 2");
    static_assert(std::numeric_limits<volatile T>::max_exponent == expected, "max_exponent test 3");
    static_assert(std::numeric_limits<const volatile T>::max_exponent == expected, "max_exponent test 4");
}

int main()
{
    test<bool, 0>();
    test<char, 0>();
    test<signed char, 0>();
    test<unsigned char, 0>();
    test<wchar_t, 0>();
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t, 0>();
    test<char32_t, 0>();
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
    test<short, 0>();
    test<unsigned short, 0>();
    test<int, 0>();
    test<unsigned int, 0>();
    test<long, 0>();
    test<unsigned long, 0>();
    test<long long, 0>();
    test<unsigned long long, 0>();
    test<float, FLT_MAX_EXP>();
    test<double, DBL_MAX_EXP>();
    test<long double, LDBL_MAX_EXP>();
}
