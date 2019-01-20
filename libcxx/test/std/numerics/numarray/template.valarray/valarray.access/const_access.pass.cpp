//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// const value_type& operator[](size_t i) const;

#include <valarray>
#include <cassert>

int main()
{
    {
        typedef int T;
        T a[] = {5, 4, 3, 2, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        const std::valarray<T> v(a, N);
        for (unsigned i = 0; i < N; ++i)
        {
            assert(v[i] == a[i]);
        }
    }
}
