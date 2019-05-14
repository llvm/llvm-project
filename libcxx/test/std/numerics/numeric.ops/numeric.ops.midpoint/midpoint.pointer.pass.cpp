//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// <numeric>

// template <class _Tp>
// _Tp* midpoint(_Tp* __a, _Tp* __b) noexcept
//

#include <numeric>
#include <cassert>

#include "test_macros.h"



template <typename T>
constexpr void constexpr_test()
{
    constexpr T array[1000] = {};
    ASSERT_SAME_TYPE(decltype(std::midpoint(array, array)), const T*);
    ASSERT_NOEXCEPT(          std::midpoint(array, array));

    static_assert(std::midpoint(array, array)        == array, "");
    static_assert(std::midpoint(array, array + 1000) == array + 500, "");

    static_assert(std::midpoint(array, array +    9) == array + 4, "");
    static_assert(std::midpoint(array, array +   10) == array + 5, "");
    static_assert(std::midpoint(array, array +   11) == array + 5, "");
    static_assert(std::midpoint(array +    9, array) == array + 5, "");
    static_assert(std::midpoint(array +   10, array) == array + 5, "");
    static_assert(std::midpoint(array +   11, array) == array + 6, "");
}

template <typename T>
void runtime_test()
{
    T array[1000] = {}; // we need an array to make valid pointers
    ASSERT_SAME_TYPE(decltype(std::midpoint(array, array)), T*);
    ASSERT_NOEXCEPT(          std::midpoint(array, array));

    assert(std::midpoint(array, array)        == array);
    assert(std::midpoint(array, array + 1000) == array + 500);

    assert(std::midpoint(array, array +    9) == array + 4);
    assert(std::midpoint(array, array +   10) == array + 5);
    assert(std::midpoint(array, array +   11) == array + 5);
    assert(std::midpoint(array +    9, array) == array + 5);
    assert(std::midpoint(array +   10, array) == array + 5);
    assert(std::midpoint(array +   11, array) == array + 6);
}

template <typename T>
void pointer_test()
{
    runtime_test<               T>();
    runtime_test<const          T>();
    runtime_test<      volatile T>();
    runtime_test<const volatile T>();

//  The constexpr tests are always const, but we can test them anyway.
    constexpr_test<               T>();
    constexpr_test<const          T>();

//  GCC 9.0.1 (unreleased as of 2019-03) barfs on this, but we have a bot for it.
//  Uncomment when gcc 9.1 is released
#ifndef TEST_COMPILER_GCC
    constexpr_test<      volatile T>();
    constexpr_test<const volatile T>();
#endif
}


int main(int, char**)
{
    pointer_test<char>();
    pointer_test<int>();
    pointer_test<double>();

    return 0;
}
