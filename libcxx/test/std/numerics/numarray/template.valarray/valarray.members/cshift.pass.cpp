//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray cshift(int i) const;

#include <valarray>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        T a2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = v1.cshift(0);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        T a2[] = {4, 5, 6, 7, 8, 9, 10, 1, 2, 3};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = v1.cshift(3);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        T a2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = v1.cshift(10);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        T a2[] = {8, 9, 10, 1, 2, 3, 4, 5, 6, 7};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = v1.cshift(17);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        T a2[] = {8, 9, 10, 1, 2, 3, 4, 5, 6, 7};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = v1.cshift(-3);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        T a2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = v1.cshift(-10);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        T a2[] = {4, 5, 6, 7, 8, 9, 10, 1, 2, 3};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = v1.cshift(-17);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }
    {
        typedef int T;
        const unsigned N1 = 0;
        std::valarray<T> v1;
        std::valarray<T> v2 = v1.cshift(-17);
        assert(v2.size() == N1);
    }
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        T a2[] = {8, 10, 12, 14, 16, 18, 20, 2, 4, 6};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = (v1 + v1).cshift(3);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        T a2[] = {16, 18, 20, 2, 4, 6, 8, 10, 12, 14};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2 = (v1 + v1).cshift(-3);
        assert(v2.size() == N1);
        for (unsigned i = 0; i < N1; ++i)
            assert(v2[i] == a2[i]);
    }

  return 0;
}
