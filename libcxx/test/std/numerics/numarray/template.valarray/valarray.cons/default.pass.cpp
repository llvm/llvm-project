//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray();

#include <valarray>
#include <cassert>

struct S {
    S() { ctor_called = true; }
    static bool ctor_called;
};

bool S::ctor_called = false;

int main(int, char**)
{
    {
        std::valarray<int> v;
        assert(v.size() == 0);
    }
    {
        std::valarray<float> v;
        assert(v.size() == 0);
    }
    {
        std::valarray<double> v;
        assert(v.size() == 0);
    }
    {
        std::valarray<std::valarray<double> > v;
        assert(v.size() == 0);
    }
    {
        std::valarray<S> v;
        assert(v.size() == 0);
        assert(!S::ctor_called);
    }

  return 0;
}
