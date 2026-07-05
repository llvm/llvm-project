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
//   operator>(const valarray<T>& x, const T& y);

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef int T;
        T a1[] = {1,  2,  3,  4,  0};
        bool a3[] = {false,  false,  true,  true,  false};
        const unsigned N = 5;
        std::valarray<T> v1(a1, N);
        std::valarray<bool> v3 = v1 > 2;
        std::valarray<bool> v3a = +(v1 > 2);
        assert(v3.size() == N);
        assert(v3a.size() == N);
        for (std::size_t i = 0; i < N; ++i) {
            assert(v3[i] == a3[i]);
            assert(v3a[i] == a3[i]);
        }
    }

  return 0;
}
