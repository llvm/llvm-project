//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <valarray>

// template<class T, size_t cnt> valarray(const T(&)[cnt], size_t) -> valarray<T>;

#include <valarray>

#include "test_macros.h"

int main(int, char**)
{
    {
        // From (initializer_list<T>)
        std::valarray v = {1, 2, 3, 4, 5};
        ASSERT_SAME_TYPE(decltype(v), std::valarray<int>);
    }

    {
        // From (const T(&)[N], size_t)
        long a[] = {1, 2, 3, 4, 5};
        std::valarray v(a, 5);
        ASSERT_SAME_TYPE(decltype(v), std::valarray<long>);
    }

    {
        // From (const T&, size_t)
        long a[] = {1, 2, 3, 4, 5};
        std::valarray v(&a[0], 5);
        // Surprising but true.
        ASSERT_SAME_TYPE(decltype(v), std::valarray<long*>);
    }

    {
        // From (slice_array<T>)
        std::valarray<long> v{1,2,3,4,5};
        std::valarray v2 = v[std::slice(2,3,1)];
        static_assert(std::is_same_v<decltype(v2), std::valarray<long>>);
    }

    {
        // From (gslice_array<T>)
        std::valarray<long> v{1,2,3,4,5};
        std::valarray v2 = v[std::gslice(0, {5}, {1})];
        static_assert(std::is_same_v<decltype(v2), std::valarray<long>>);
    }

    {
        // From (mask_array<T>)
        std::valarray<long> v = {1, 2, 3, 4, 5};
        std::valarray<bool> m = {true, false, true, false, true};
        std::valarray v2 = v[m];
        static_assert(std::is_same_v<decltype(v2), std::valarray<long>>);
    }

    {
        // From (indirect_array<T>)
        std::valarray<long> v = {1, 2, 3, 4, 5};
        std::valarray<size_t> i = {1, 2, 3};
        std::valarray v2 = v[i];
        static_assert(std::is_same_v<decltype(v2), std::valarray<long>>);
    }

  return 0;
}
