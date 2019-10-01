//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// explicit valarray(size_t);

#include <valarray>
#include <cassert>

#include "test_macros.h"

struct S {
    S() : x(1) {}
    ~S() { ++cnt_dtor; }
    int x;
    static size_t cnt_dtor;
};

size_t S::cnt_dtor = 0;

int main(int, char**)
{
    {
        std::valarray<int> v(100);
        assert(v.size() == 100);
        for (int i = 0; i < 100; ++i)
            assert(v[i] == 0);
    }
    {
        std::valarray<double> v(100);
        assert(v.size() == 100);
        for (int i = 0; i < 100; ++i)
            assert(v[i] == 0);
    }
    {
        std::valarray<std::valarray<double> > v(100);
        assert(v.size() == 100);
        for (int i = 0; i < 100; ++i)
            assert(v[i].size() == 0);
    }
    {
        std::valarray<S> v(100);
        assert(v.size() == 100);
        for (int i = 0; i < 100; ++i)
            assert(v[i].x == 1);
    }
    assert(S::cnt_dtor == 100);

  return 0;
}
