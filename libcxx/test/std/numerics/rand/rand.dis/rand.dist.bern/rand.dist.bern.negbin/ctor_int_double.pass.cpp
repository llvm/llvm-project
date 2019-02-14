//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class negative_binomial_distribution

// explicit negative_binomial_distribution(IntType t = 1, double p = 0.5);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::negative_binomial_distribution<> D;
        D d;
        assert(d.k() == 1);
        assert(d.p() == 0.5);
    }
    {
        typedef std::negative_binomial_distribution<> D;
        D d(3);
        assert(d.k() == 3);
        assert(d.p() == 0.5);
    }
    {
        typedef std::negative_binomial_distribution<> D;
        D d(3, 0.75);
        assert(d.k() == 3);
        assert(d.p() == 0.75);
    }

  return 0;
}
