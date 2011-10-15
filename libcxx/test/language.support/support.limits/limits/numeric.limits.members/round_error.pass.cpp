//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// round_error()

#include <limits>
#include <cfloat>
#include <cassert>

template <class T>
void
test(T expected)
{
    assert(std::numeric_limits<T>::round_error() == expected);
    assert(std::numeric_limits<const T>::round_error() == expected);
    assert(std::numeric_limits<volatile T>::round_error() == expected);
    assert(std::numeric_limits<const volatile T>::round_error() == expected);
}

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
    test<float>(0.5);
    test<double>(0.5);
    test<long double>(0.5);
}
