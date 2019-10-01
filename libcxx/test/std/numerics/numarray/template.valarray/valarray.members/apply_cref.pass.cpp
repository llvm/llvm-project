//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray apply(value_type f(const value_type&)) const;

#include <valarray>
#include <cassert>

#include "test_macros.h"

typedef int T;

T f(const T& t) {return t + 5;}

int main(int, char**)
{
    {
        T a1[] = {1, 2, 3, 4,  5,  6,  7,  8,  9, 10};
        T a2[] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = v1.apply(f);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }
    {
        const unsigned N1 = 0;
        std::valarray<T> v1;
        std::valarray<T> v2 = v1.apply(f);
        assert(v2.size() == N1);
    }
    {
        T a1[] = {1, 2, 3, 4,  5,  6,  7,  8,  9, 10};
        T a2[] = {7, 9, 11, 13, 15, 17, 19, 21, 23, 25};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = (v1+v1).apply(f);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }

  return 0;
}
