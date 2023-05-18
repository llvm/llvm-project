//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray operator+() const;

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef int T;
        T a[] = {1, 2, 3, 4, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2 = +v;
        assert(v2.size() == v.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == +v[i]);
    }
    {
        typedef double T;
        T a[] = {1, 2.5, 3, 4.25, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2 = +v;
        assert(v2.size() == v.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == +v[i]);
    }
    {
        typedef std::valarray<double> T;
        T a[] = {T(1), T(2), T(3), T(4), T(5)};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2 = +v;
        assert(v2.size() == v.size());
        for (unsigned i = 0; i < N; ++i)
        {
            assert(v2[i].size() == v[i].size());
            for (std::size_t j = 0; j < v[i].size(); ++j)
                assert(v2[i][j] == +v[i][j]);
        }
    }
    {
        typedef double T;
        T a[] = {1, 2.5, 3, 4.25, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2 = +(v + v);
        assert(v2.size() == v.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == +2*v[i]);
    }
    {
        // Make sure we don't have dangling reference problems with unary expressions
        int array[] = {1, 2, 3};
        std::valarray<int> a(array, 3);
        std::valarray<int> b(array, 3);
        auto c = +a * b;
        assert(c.size() == 3);
        assert(c[0] == 1 && c[1] == 4 && c[2] == 9);
    }

    return 0;
}
