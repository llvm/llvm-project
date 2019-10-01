//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// template <class T>
//   unspecified1
//   end(const valarray<T>& v);

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
        const std::valarray<T> v(a, N);
        assert(v[v.size()-1] == 5);
        assert(static_cast<std::size_t>(end(v) - begin(v)) == v.size());
    }

  return 0;
}
