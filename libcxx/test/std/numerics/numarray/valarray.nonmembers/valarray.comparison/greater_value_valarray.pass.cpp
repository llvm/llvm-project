//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// template<class T>
//   valarray<bool>
//   operator>(const T& x, const valarray<T>& y);

#include <valarray>
#include <cassert>
#include <cstddef>

int main()
{
    {
        typedef int T;
        T a2[] = {1,  2,  3,  4,  0};
        bool a3[] = {true,  false,  false,  false,  true};
        const unsigned N = sizeof(a2)/sizeof(a2[0]);
        std::valarray<T> v2(a2, N);
        std::valarray<bool> v3 = 2 > v2;
        assert(v2.size() == v3.size());
        for (std::size_t i = 0; i < v3.size(); ++i)
            assert(v3[i] == a3[i]);
    }
}
