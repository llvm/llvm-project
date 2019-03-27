//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray(const value_type& x, size_t n);

#include <valarray>
#include <cassert>

int main(int, char**)
{
    {
        std::valarray<int> v(5, 100);
        assert(v.size() == 100);
        for (int i = 0; i < 100; ++i)
            assert(v[i] == 5);
    }
    {
        std::valarray<double> v(2.5, 100);
        assert(v.size() == 100);
        for (int i = 0; i < 100; ++i)
            assert(v[i] == 2.5);
    }
    {
        std::valarray<std::valarray<double> > v(std::valarray<double>(10), 100);
        assert(v.size() == 100);
        for (int i = 0; i < 100; ++i)
            assert(v[i].size() == 10);
    }

  return 0;
}
