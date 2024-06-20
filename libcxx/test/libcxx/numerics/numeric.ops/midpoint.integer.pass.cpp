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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "test_macros.h"

//  Users are not supposed to provide template argument lists for
//  functions in the standard library (there's an exception for min and max)
//  However, libc++ protects against this for pointers. The use of T(0)
//  in the test cases resolves potential ambiguity in template argument deduction
//  for the std::midpoint function.

template <typename T>
void test()
{
  ASSERT_SAME_TYPE(T, decltype(std::midpoint<T>(T(0), T(0))));
}

int main(int, char**)
{
    test<signed char>();
    test<short>();
    test<int>();
    test<long>();
    test<long long>();

    test<std::int8_t>();
    test<std::int16_t>();
    test<std::int32_t>();
    test<std::int64_t>();

    test<unsigned char>();
    test<unsigned short>();
    test<unsigned int>();
    test<unsigned long>();
    test<unsigned long long>();

    test<std::uint8_t>();
    test<std::uint16_t>();
    test<std::uint32_t>();
    test<std::uint64_t>();

#ifndef TEST_HAS_NO_INT128
    test<__int128_t>();
    test<__uint128_t>();
#endif

    test<char>();
    test<std::ptrdiff_t>();
    test<std::size_t>();

    return 0;
}
