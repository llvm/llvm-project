//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray<bool> operator!() const;

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
        std::valarray<bool> v2 = !v;
        assert(v2.size() == v.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == !v[i]);
    }
    {
        typedef double T;
        T a[] = {1, 2.5, 3, 4.25, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<bool> v2 = !(v + v);
        assert(v2.size() == v.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == !(2 * v[i]));
    }

  return 0;
}
