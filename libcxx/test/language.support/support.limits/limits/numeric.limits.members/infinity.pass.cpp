//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// infinity()

#include <limits>
#include <cfloat>
#include <cassert>

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

int main()
{
    test<bool>(false);
    test<char>(0);
    test<signed char>(0);
    test<unsigned char>(0);
    test<wchar_t>(0);
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t>(0);
    test<char32_t>(0);
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
    test<short>(0);
    test<unsigned short>(0);
    test<int>(0);
    test<unsigned int>(0);
    test<long>(0);
    test<unsigned long>(0);
    test<long long>(0);
    test<unsigned long long>(0);
    test<float>(1./zero);
    test<double>(1./zero);
    test<long double>(1./zero);
}

float zero = 0;
