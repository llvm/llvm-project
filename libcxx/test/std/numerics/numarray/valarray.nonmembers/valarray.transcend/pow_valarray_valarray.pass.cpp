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
//   pow(const valarray<T>& x, const valarray<T>& y);

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "valarray_helper.h"

int main(int, char**)
{
    {
        typedef double T;
        T a1[] = {.9, .5, 0., .5, .75};
        T a2[] = {-.8,  .25, 0.375, -.5, .75};
        T a3[] = {1.0879426248455297e+00,
                  8.4089641525371450e-01,
                  0.0000000000000000e+00,
                  1.4142135623730949e+00,
                  8.0592744886765644e-01};
        const unsigned N = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N);
        std::valarray<T> v2(a2, N);
        std::valarray<T> v3 = pow(v1, v2);
        assert(v3.size() == v1.size());
        for (std::size_t i = 0; i < v3.size(); ++i)
            assert(is_about(v3[i], a3[i], 10));
    }

  return 0;
}
