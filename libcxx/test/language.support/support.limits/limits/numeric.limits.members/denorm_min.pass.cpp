//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// denorm_min()

#include <limits>
#include <cassert>

template <class T>
void
test(T expected)
{
    assert(std::numeric_limits<T>::denorm_min() == expected);
    assert(std::numeric_limits<const T>::denorm_min() == expected);
    assert(std::numeric_limits<volatile T>::denorm_min() == expected);
    assert(std::numeric_limits<const volatile T>::denorm_min() == expected);
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
    test<float>(__FLT_DENORM_MIN__);
    test<double>(__DBL_DENORM_MIN__);
    test<long double>(__LDBL_DENORM_MIN__);
}
