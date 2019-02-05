//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class discrete_distribution

// result_type max() const;

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::discrete_distribution<> D;
        double p0[] = {.3, .1, .6};
        D d(p0, p0+3);
        assert(d.max() == 2);
    }
    {
        typedef std::discrete_distribution<> D;
        double p0[] = {.3, .1, .6, .2};
        D d(p0, p0+4);
        assert(d.max() == 3);
    }
}
