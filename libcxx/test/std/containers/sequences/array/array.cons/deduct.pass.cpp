//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>
// UNSUPPORTED: c++03, c++11, c++14

// template <class T, class... U>
//   array(T, U...) -> array<T, 1 + sizeof...(U)>;
//
//  Requires: (is_same_v<T, U> && ...) is true. Otherwise the program is ill-formed.

#include <array>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

constexpr bool tests()
{
    //  Test the explicit deduction guides
    {
        std::array arr{1,2,3};  // array(T, U...)
        static_assert(std::is_same_v<decltype(arr), std::array<int, 3>>, "");
        assert(arr[0] == 1);
        assert(arr[1] == 2);
        assert(arr[2] == 3);
    }

    {
        const long l1 = 42;
        std::array arr{1L, 4L, 9L, l1}; // array(T, U...)
        static_assert(std::is_same_v<decltype(arr)::value_type, long>, "");
        static_assert(arr.size() == 4, "");
        assert(arr[0] == 1);
        assert(arr[1] == 4);
        assert(arr[2] == 9);
        assert(arr[3] == l1);
    }

    //  Test the implicit deduction guides
    {
        std::array<double, 2> source = {4.0, 5.0};
        std::array arr(source);   // array(array)
        static_assert(std::is_same_v<decltype(arr), decltype(source)>, "");
        static_assert(std::is_same_v<decltype(arr), std::array<double, 2>>, "");
        assert(arr[0] == 4.0);
        assert(arr[1] == 5.0);
    }

    return true;
}

int main(int, char**)
{
    tests();
    static_assert(tests(), "");
    return 0;
}
