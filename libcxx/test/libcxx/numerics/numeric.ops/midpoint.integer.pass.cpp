//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// <numeric>

// template <class _Tp>
// _Tp midpoint(_Tp __a, _Tp __b) noexcept
//

#include <numeric>
#include <cassert>
#include "test_macros.h"

//  Users are not supposed to provide template argument lists for
//  functions in the standard library (there's an exception for min and max)
//  However, libc++ protects against this for pointers, so we check to make
//  sure that our protection is working here.
//  In some cases midpoint<int>(0,0) might get deduced as the pointer overload.

template <typename T>
void test()
{
    ASSERT_SAME_TYPE(T, decltype(std::midpoint<T>(0, 0)));
}

int main(int, char**)
{
    test<signed char>();
    test<short>();
    test<int>();
    test<long>();
    test<long long>();

    test<int8_t>();
    test<int16_t>();
    test<int32_t>();
    test<int64_t>();

    test<unsigned char>();
    test<unsigned short>();
    test<unsigned int>();
    test<unsigned long>();
    test<unsigned long long>();

    test<uint8_t>();
    test<uint16_t>();
    test<uint32_t>();
    test<uint64_t>();

#ifndef TEST_HAS_NO_INT128
    test<__int128_t>();
    test<__uint128_t>();
#endif

    test<char>();
    test<ptrdiff_t>();
    test<size_t>();

    return 0;
}
