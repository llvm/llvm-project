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
//   valarray<T>
//   log10(const valarray<T>& x);

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "valarray_helper.h"

int main(int, char**)
{
    {
        typedef double T;
        T a1[] = {.5, .75, 1, 3, 7};
        T a3[] = {-3.0102999566398120e-01,
                  -1.2493873660829995e-01,
                   0.0000000000000000e+00,
                   4.7712125471966244e-01,
                   8.4509804001425681e-01};
        const unsigned N = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N);
        std::valarray<T> v3 = log10(v1);
        assert(v3.size() == v1.size());
        for (std::size_t i = 0; i < v3.size(); ++i)
            assert(is_about(v3[i], a3[i], 10));
    }

  return 0;
}
