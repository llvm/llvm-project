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
void pointer_test()
{
    T array[1000] = {}; // we need an array to make valid pointers
    constexpr T cArray[2] = {};
    ASSERT_SAME_TYPE(decltype(std::midpoint(array, array)), T*);
    ASSERT_NOEXCEPT(          std::midpoint(array, array));

    static_assert(std::midpoint(cArray, cArray + 2) == cArray + 1, "");
    static_assert(std::midpoint(cArray + 2, cArray) == cArray + 1, "");

    assert(std::midpoint(array, array)        == array);
    assert(std::midpoint(array, array + 1000) == array + 500);

    assert(std::midpoint(array, array +    9) == array + 4);
    assert(std::midpoint(array, array +   10) == array + 5);
    assert(std::midpoint(array, array +   11) == array + 5);
    assert(std::midpoint(array +    9, array) == array + 5);
    assert(std::midpoint(array +   10, array) == array + 5);
    assert(std::midpoint(array +   11, array) == array + 6);
}


int main(int, char**)
{
    pointer_test<               char>();
    pointer_test<const          char>();
    pointer_test<      volatile char>();
    pointer_test<const volatile char>();
    
    pointer_test<int>();
    pointer_test<double>();

    return 0;
}
